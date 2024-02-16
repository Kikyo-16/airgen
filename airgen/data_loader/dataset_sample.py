import math
from torch.utils.data import Dataset as BaseDataset
from ..utilities import *
import scipy.signal as signal


def merge(seq, cond, st, ed, k=10):
    res = np.concatenate([
        cond[:, st: ed],
        seq[:, st: ed]], -1)
    ed = ed - st
    mask = np.zeros_like(res[0])
    mask[ed + k:] = 1
    return res, mask


def sample_mask(mix, cond, onset=0):
    k = 5
    for i in range(k):
        mix[:, onset + i] = np.arange(4)
        cond[:, onset + i] = 3 - np.arange(4)
    return mix, cond


def load_data_from_path(tag, path, idx, sec):
    with open(path, "r") as f:
        lines = f.readlines()
    data = []
    data_onset = []
    flen = 0
    for i, line in enumerate(lines):
        line = line.rstrip()
        f_path = line.split("\t\t")[0]
        onset = float(line.split("\t\t")[1])
        offset = float(line.split("\t\t")[2])
        data += [{"path": f_path,
                  "data": {
                      "mix":
                          np.load(os.path.join(f_path, "mix_rvq.npy")),
                      "piano":
                          np.load(os.path.join(f_path, f"{tag}_rvq.npy"))
                  }}]

        x_len = min(data[i]["data"]["mix"].shape[-1], data[i]["data"]["piano"].shape[-1]) / 50
        if offset == -1 or x_len < offset:
            offset = x_len
        onset = math.ceil(onset)
        offset = int(offset)
        data_onset.append([idx, i, onset, offset - sec])
        flen += (onset + offset - sec) / 10.

    return data, data_onset, flen


def get_prompt(trk_id):
    if trk_id == 0:
        return "drums loop"
    if trk_id == 1:
        return "generate music: "


class Dataset(BaseDataset):
    def __init__(self, path_lst, cfg, rid, inference=False, tag="drums"):
        super(Dataset, self).__init__()
        self.rid = rid
        self.rng = np.random.RandomState(42 + rid * 100)
        self.cfg = cfg
        self.data = []
        self.data_onset = []
        flen = 0

        for i, path in enumerate(path_lst):
            data, data_onset, fl = load_data_from_path(tag, path, i, cfg.sample_sec)
            self.data.append(data)
            self.data_onset += data_onset
            flen += fl

        self.f_len = int(flen)

        print("num of files", self.f_len)

        self.epoch = 0
        self.f_offset = 0
        self.inference = inference

        self.descs = [
            "catchy song",
            "melodic music piece",
            "a song",
            "music tracks",
        ]

    def load_data(self, set_id, song_id):
        data = self.data[set_id]
        if "piano" not in data[song_id]["data"]:
            mix = data[song_id]["data"]["mix"]
            piano = data[song_id]["data"]["piano"]

            result = {
                "piano": piano,
                "mix": mix,
            }
            data[song_id]["data"] = result
        return data[song_id]["data"]

    def __len__(self):
        return self.f_len

    def sample(self, idx, frame):
        idx = int(idx % len(self.data_index))
        set_id, sid, sec_id = self.data_index[idx]
        data = self.load_data(set_id, sid)

        st = sec_id
        res = self.cfg.frame_res
        frame_st = int(st * res)
        frame_ed = int(frame_st + frame)

        cond = data["piano"]
        mix = data["mix"]

        r = .8
        dt = self.rng.rand() * (r - .4) + .4
        mix = mix[:, frame_st: frame_ed]
        cond = cond[:, frame_st: frame_ed] + 0

        mask = self.rng.rand(cond.shape[-1])
        win_len = 11
        mask[mask < dt] = 0
        mask[mask > 0] = 1
        mask = signal.medfilt(mask, win_len)
        ori = mask == 1
        cond[:, ori] = mix[:, ori]
        ind = np.concatenate([mask, mask + 2, mask[-1:] + 2], -1)
        return mix, cond, ind

    def __getitem__(self, idx):
        sample_sec = self.cfg.sample_sec
        sample_frame = int(sample_sec * self.cfg.frame_res)

        mix, cond, ind = self.sample(idx, sample_frame // 2)
        seq = np.concatenate([cond, mix], -1)

        return {
            "seq": seq,
            "mask": ind,
            "desc": None,
        }

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e
        self.rng.shuffle(self.data_onset)
        data_index = []
        for i, j, st, ed in self.data_onset:
            data_index += [[i, j, k] for k in range(st + self.rng.randint(0, 10), ed, 10)]
        self.data_index = data_index


def collate_fn(batch):
    seq = torch.stack([torch.from_numpy(d["seq"]) for d in batch], 0)
    mask = torch.stack([torch.from_numpy(d["mask"]) for d in batch], 0)
    desc = [d["desc"] for d in batch]
    return {
        "seq": seq,
        "mask": mask,
        "desc": desc,
    }
