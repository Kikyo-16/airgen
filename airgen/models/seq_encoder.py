import torch
import torch.nn as nn


class SeqEncoder(nn.ModuleList):
    def __init__(self, output_dim=512, tokenizer=None, encoder=None):
        super(SeqEncoder, self).__init__()
        name = "flan-t5-base"
        if tokenizer is None:
            self.t5 = {
                "tokenizer": T5Tokenizer.from_pretrained(name),
                "encoder": T5EncoderModel.from_pretrained(name).train(mode=False)
            }
        else:
            self.t5 = {
                "tokenizer": tokenizer,
                "encoder": encoder
            }
        self.output_proj = nn.Linear(hidden_size, output_dim=output_dim)

    def tokenize(self, x):
        entries = [xi if xi is not None else "" for xi in x]
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])
        inputs = self.t5["tokenizer"](entries, return_tensors="pt", padding=True).to(self.device)
        mask = inputs["attention_mask"]
        mask[empty_idx, :] = 0
        return inputs

    def forward(self, inputs):
        mask = inputs["attention_mask"]
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5["encoder"](**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1))
        out = self.output_proj(embeds)
        return out, mask
