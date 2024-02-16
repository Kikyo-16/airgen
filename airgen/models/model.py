import torch
from torch import nn
from .musicgen_air import MusicGen
from ..utilities.model_utils import freeze, print_trainable_parameters


def get_musicgen(sec, device):
    mg = MusicGen.get_pretrained(name='large', device=device)
    mg.set_generation_params(duration=sec, extend_stride=16, top_k=250)
    freeze(mg.lm)
    return mg


def tile_list(x):
    return [torch.cat([v, v], 0) for v in x]


class CondMusicgen(nn.Module):
    def __init__(self, sec, device="cuda", frame_rate=50):
        super().__init__()
        mg = get_musicgen(sec, device)
        self.musicgen = mg
        self.lm = mg.lm
        self.max_duration = sec
        self.frame_rate = frame_rate

    def set_training(self):
        self.lm.train()

    def forward(self, seq, desc, embed_fn, num_samples=1, mode="train",
                total_gen_len=None, prompt_tokens=None):

        mg = self.musicgen
        lm = self.lm

        attributes, _ = mg._prepare_tokens_and_attributes(desc, None)

        if mode == "train":
            with mg.autocast:
                out = lm.compute_predictions(codes=seq,
                                             embed_fn=embed_fn,
                                             conditions=attributes)
            return out
        elif mode == "inference":
            if total_gen_len is None:
                total_gen_len = int(mg.duration * mg.frame_rate)

            with mg.autocast:
                gen_tokens = lm.generate(embed_fn=embed_fn, num_samples=num_samples,
                                         prompt=None, conditions=attributes,
                                         callback=None, max_gen_len=total_gen_len, **mg.generation_params)
                return gen_tokens
        elif mode == "continuation":
            with mg.autocast:
                # if prompt_tokens is not None:
                #    print(prompt_tokens.shape)
                gen_tokens = lm.generate(embed_fn=embed_fn, num_samples=num_samples,
                                         prompt=prompt_tokens, conditions=attributes,
                                         callback=None, max_gen_len=total_gen_len, **mg.generation_params)
                return gen_tokens

    def generate(self, fn, desc, prompt, xlen):

        mg = self.musicgen
        lm = self.lm

        attributes, _ = mg._prepare_tokens_and_attributes(desc, None)

        all_tokens = []
        stride_tokens = int(self.frame_rate * mg.extend_stride)
        current_gen_offset = 0
        prompt_length = prompt.shape[-1]
        prompt_tokens = prompt
        total_gen_len = prompt_length + xlen
        total_sec = total_gen_len / 50.
        #print(current_gen_offset, prompt_length, total_gen_len)
        while current_gen_offset + prompt_length < total_gen_len:
            time_offset = current_gen_offset / self.frame_rate
            chunk_duration = min(total_sec - time_offset, self.max_duration)
            max_gen_len = int(chunk_duration * self.frame_rate)
            #print(max_gen_len, prompt_length)
            if prompt_length >= max_gen_len:
                break
            # print("current_gen_offset / total ", current_gen_offset, "/", total_gen_len)
            with mg.autocast:
                gen_tokens = lm.generate(num_samples=1,
                                         embed_fn=fn, prompt=prompt_tokens,
                                         conditions=attributes,
                                         callback=None, max_gen_len=max_gen_len, **mg.generation_params)
            if prompt_tokens is None:
                all_tokens.append(gen_tokens)
            else:
                all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
            prompt_tokens = gen_tokens[:, :, stride_tokens:]
            prompt_length = prompt_tokens.shape[-1]
            current_gen_offset += stride_tokens
            if current_gen_offset > 50 * 80:
                break

        gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

    def get_input_embeddings(self):
        return self.lm.emb


class EmbFn:
    def __init__(self, fn, trk_id, start_layer, max_len, inference=False):
        self.interval = None
        self.index = -1
        self.adaptor = None
        self.start_layer = start_layer
        self.max_len = max_len
        self.fn = fn
        self.trk_id = trk_id
        self.inference = inference


    def apply_adaptor(self, tag, x, q, p):
        index = self.index
        if index < self.start_layer or tag == "cross":
            return x
        i = index - self.start_layer
        if self.inference:
            st, ed = self.interval

            trk_id = self.trk_id[:, st:ed]
        else:
            trk_id = self.trk_id

        x = self.fn(i, x, q, p, trk_id=trk_id)
        return x

    def get_cross_attention_src(self, src):
        return src

    def crop(self, x):

        if self.interval is not None:
            st, ed = self.interval
            if st >= self.max_len:
                st = self.max_len - 1
                ed = st + 1
            return x[:, :, st:ed, :]
        return x

    def update_adaptor(self, adaptor):
        self.adaptor = adaptor


    def set_index(self, index):
        self.index = index

    def update_interval(self, st, ed):
        self.interval = [st, ed]

    def clear_state(self):
        todo = 0



class CPTransformerLayer(nn.Module):
    def __init__(self, norm1, norm2, layer_scale_1, dropout1, self_attn, layer_scale_2,
                 autocast, linear1, linear2, activation, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = norm1
        self.norm2 = norm2
        self.layer_scale_1 = layer_scale_1
        self.dropout1 = dropout1
        self.self_attn = self_attn
        self.layer_scale_2 = layer_scale_2
        self.autocast = autocast
        self.linear1 = linear1
        self.linear2 = linear2
        self.activation = activation
        self.dropout = dropout

    def _ff_block(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def forward(self, x):
        with self.autocast:
            nx = self.norm1(x)
            q, k, v, o = self.self_attn(nx, nx, nx, emb_fn=None,
                                        attn_mask=None,
                                        key_padding_mask=None,
                                        need_weights=False, is_causal=False, return_qkv=True)
            x = x + self.layer_scale_1(self.dropout1(o))
            x = x + self.layer_scale_2(self._ff_block(self.norm2(x)))
        return q, k, v, x


class ICLAdaptor(nn.Module):
    def __init__(self, hidden_dim, start_layer, n_layers, model, autocast, n_tasks, k):
        super().__init__()
        self.adaptor = nn.Parameter(
            torch.randn(n_layers, n_tasks, k, hidden_dim),
            requires_grad=True)
        self.gates = nn.Parameter(
            torch.zeros([n_layers, n_tasks]),
            requires_grad=True)

        new_layers = nn.ModuleList()
        max_n_frames = 1000

        for i in range(start_layer, len(model.layers)):
            norm1 = model.layers[i].norm1
            norm2 = model.layers[i].norm2
            layer_scale_1 = model.layers[i].layer_scale_1
            dropout1 = model.layers[i].dropout1
            self_attn = model.layers[i].self_attn
            layer_scale_2 = model.layers[i].layer_scale_2
            linear1 = model.layers[i].linear1
            linear2 = model.layers[i].linear2
            activation = model.layers[i].activation
            dropout = model.layers[i].dropout
            new_layers.append(CPTransformerLayer(norm1=norm1,
                                                 norm2=norm2,
                                                 layer_scale_1=layer_scale_1,
                                                 dropout1=dropout1,
                                                 self_attn=self_attn,
                                                 linear1=linear1,
                                                 linear2=linear2,
                                                 activation=activation,
                                                 dropout=dropout,
                                                 layer_scale_2=layer_scale_2,
                                                 autocast=autocast))
        self.layers = new_layers
        self.activates_dict = {}

        freeze(self.layers)

        self.max_n_frames = max_n_frames
        self.start_layer = start_layer
        self.num_layers = n_layers
        self.n_tasks = n_tasks

    def fn(self, i, x, q, p, trk_id):
        k, v, g = self.activates_dict[str(i)]
        b, d1, t, d2 = q.shape
        q = q.permute(0, 2, 1, 3).flatten(0, 1)
        dx = torch.zeros_like(q)

        for i in range(len(k)):
            idx = (trk_id == i).flatten()
            if idx.sum() == 0:
                continue
            qt = q[idx].view(1, -1, d1, d2).permute(0, 2, 1, 3)
            dx_t = torch.nn.functional.scaled_dot_product_attention(
                qt, k[i, None], v[i, None], is_causal=False, dropout_p=p)*g[i]
            dx[idx] = dx_t.permute(0, 2, 1, 3).flatten(0, 1)
        dx = dx.view(b, t, d1, d2).permute(0, 2, 1, 3)
        return x + dx

    def forward(self, trk_id, inference=False):
        for i in range(self.num_layers):

            _, k, v, _ = self.layers[i](self.adaptor[i])

            self.activates_dict[str(i)] = [k, v, self.gates[i]]

        emb_fn = EmbFn(inference=inference,
                       fn=self.fn, trk_id=trk_id,
                       start_layer=self.start_layer,
                       max_len=self.max_n_frames)

        return emb_fn

    def save_weights(self, path):
        state_dict = {}
        sdict = self.state_dict()
        for n in sdict:
            if str.startswith(n, "layers"):
                continue
            state_dict[n] = sdict[n]
        torch.save(state_dict, path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)


class AIRGen(nn.Module):
    def __init__(self, sec, num_layers, n_tasks, k):
        super().__init__()
        lm = CondMusicgen(sec)
        self.peft_model = lm
        self.musicgen = lm.musicgen
        self.icl_adaptor = ICLAdaptor(model=self.musicgen.lm.transformer,
                                      start_layer=48 - num_layers,
                                      hidden_dim=2048,
                                      n_layers=num_layers,
                                      autocast=self.musicgen.autocast,
                                      n_tasks=n_tasks,
                                      k=k)

    def set_training(self):
        self.peft_model.set_training()
        print_trainable_parameters(self)

    def save_weights(self, path):
        self.icl_adaptor.save_weights(path)

    def load_weights(self, path):
        self.icl_adaptor.load_weights(path)

    def forward(self, seq, desc, trk_id, num_samples=None, mode="train",
                max_n_frames=None, prompt_tokens=None):
        embed_fn = self.icl_adaptor(trk_id)
        out = self.peft_model(seq, desc=desc, embed_fn=embed_fn,
                              mode=mode, num_samples=num_samples, total_gen_len=max_n_frames,
                              prompt_tokens=prompt_tokens)
        return out

    def generate(self, prompt, desc, trk_id, xlen):
        trk_id = torch.cat([trk_id, trk_id], 0)
        emb_fn = self.icl_adaptor(trk_id, inference=True)
        out = self.peft_model.generate(fn=emb_fn, desc=desc, prompt=prompt, xlen=xlen)
        return out
