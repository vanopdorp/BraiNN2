#%%writefile model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
import faiss

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODEL_DIM = 8192
SPARSITY = 0.01
K = max(1, int(MODEL_DIM * SPARSITY))
GATE_RANK = 4
LR = 1e-3


class BioBrainLayer(nn.Module):
    def __init__(self, dim, k, lr=1e-3):
        super().__init__()
        self.dim = dim
        self.k = k
        self.lr = lr

        self.weight = nn.Parameter(
            torch.randn(dim, dim, device=DEVICE, dtype=torch.float16) * 0.01
        )

        self.register_buffer("prev_vals", torch.zeros(1, dim, dtype=torch.float16))
        self.register_buffer("prev_idx",  torch.zeros(1, dtype=torch.long))

        self.register_buffer("usage",      torch.zeros(dim, dtype=torch.float16))
        self.register_buffer("prev_topk",  torch.zeros(k,  dtype=torch.long))
    
        self.register_buffer("opt_momentum", torch.zeros_like(self.weight))
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.clip_norm = 1.0
    @torch.no_grad()
    def optimizer_step(self, grad, active_idx=None):
        grad = grad - self.weight_decay * self.weight

        gnorm = grad.norm()
        if gnorm > self.clip_norm:
            grad = grad * (self.clip_norm / (gnorm + 1e-8))

        self.opt_momentum = self.momentum * self.opt_momentum + (1 - self.momentum) * grad

        if active_idx is not None:
            self.weight.data[:, active_idx] -= self.lr * self.opt_momentum[:, active_idx]
        else:
            self.weight.data -= self.lr * self.opt_momentum

        self.weight.data = F.normalize(self.weight.data, dim=0)

    @torch.no_grad()
    def forward(self, active_vals, active_idx):
        active_vals = active_vals.to(DEVICE).half()
        active_idx  = active_idx.to(DEVICE).long()

        w_sub = self.weight[:, active_idx]
        out   = torch.matmul(active_vals, w_sub.t())
        out   = F.relu(out)

        norm_out = F.normalize(out, dim=1)
        scores   = norm_out.mean(dim=0)

        T = 0.07
        scores = scores / T

        noise_thresh = scores.mean() - 0.5 * scores.std()
        scores = torch.where(scores < noise_thresh, torch.zeros_like(scores), scores)

        self.usage = 0.99 * self.usage + 0.01 * (scores > 0).float()
        scores = scores - 0.1 * self.usage

        pre_topk = torch.topk(scores, self.k).indices
        inhib_mask = torch.ones_like(scores, dtype=torch.bool)
        inhib_mask[pre_topk] = False
        scores = scores - 0.2 * inhib_mask.float() * scores

        prev_mask = torch.zeros_like(scores)
        prev_mask[self.prev_topk] = 0.1
        scores = scores + prev_mask

        topk = torch.topk(scores, self.k).indices
        self.prev_topk = topk.clone()

        mask = torch.zeros_like(scores)
        mask[topk] = 1.0

        out_k = out[:, topk]

        self.prev_vals = active_vals.clone()
        self.prev_idx  = active_idx.clone()

        return out_k, topk


    @torch.no_grad()
    def predictive_update(self, x_real, x_pred, active_vals, active_idx):
        x_real = torch.nan_to_num(x_real, nan=0.0)
        x_pred = torch.nan_to_num(x_pred, nan=0.0)
        prev    = torch.nan_to_num(self.prev_vals, nan=0.0)

        error = (x_pred - x_real)
        grad  = torch.matmul(error.t(), prev)

        self.optimizer_step(grad, active_idx)


    @torch.no_grad()
    def predict(self, active_vals, active_idx):

        active_vals = active_vals.to(DEVICE).half()
        active_idx = active_idx.to(DEVICE).long()

        w_sub = self.weight[:, active_idx]
        pred = torch.matmul(active_vals, w_sub.t())
        return pred
    @torch.no_grad()
    def contrastive_update(self, anchor, positive, active_idx=None):
        T = 0.07

        anchor   = F.normalize(torch.nan_to_num(anchor), dim=-1)
        positive = F.normalize(torch.nan_to_num(positive), dim=-1)

        if active_idx is not None:
            all_idx = torch.arange(self.dim, device=DEVICE)
            mask = torch.ones_like(all_idx, dtype=torch.bool)
            mask[active_idx] = False
            neg_idx = all_idx[mask]
            n = F.normalize(self.weight[:, neg_idx].mean(dim=1), dim=0)
        else:
            n = F.normalize(self.weight.mean(dim=1), dim=0)

        sim_pos = (anchor * positive).sum() / T
        sim_neg = (anchor * n).sum() / T

        error = torch.tanh(sim_pos - sim_neg)
        grad  = error * anchor.unsqueeze(1)

        self.optimizer_step(grad, active_idx)

class AutoencoderLayer(nn.Module):
    def __init__(self, dim, k, lr=1e-3):
        super().__init__()
        self.encoder = BioBrainLayer(dim, k, lr=lr)
        self.decoder = BioBrainLayer(dim, k, lr=lr)
        self.dim = dim
        self.k = k
        self.lr = lr

    @torch.no_grad()
    def forward(self, x, do_update=True):
        B, D = x.shape
        x = x.to(DEVICE).half()

        active_idx = torch.nonzero(x[0] != 0, as_tuple=False).squeeze(1)
        if active_idx.numel() == 0:
            active_idx = torch.arange(D, device=DEVICE, dtype=torch.long)
        active_idx = torch.clamp(active_idx, 0, D - 1)

        enc_input = x[:, active_idx]
        enc_vals, enc_idx = self.encoder(enc_input, active_idx)

        dec_input = enc_vals
        dec_vals, dec_idx = self.decoder(dec_input, enc_idx)

        recon = torch.zeros(B, self.dim, device=DEVICE, dtype=torch.float16)
        recon[:, dec_idx] = dec_vals

        if do_update:


            enc_pred = self.encoder.predict(enc_input, active_idx)

            self.encoder.predictive_update(
                x_real=enc_input,
                x_pred=enc_pred,
                active_vals=enc_input,
                active_idx=active_idx
            )


            dec_pred = self.decoder.predict(dec_input, enc_idx)

            self.decoder.predictive_update(
                x_real=x[:, enc_idx],
                x_pred=dec_pred,
                active_vals=dec_input,
                active_idx=enc_idx
            )


            anchor   = enc_vals.mean(dim=0)
            positive = dec_vals.mean(dim=0)
            negative = torch.randn_like(positive)

            self.encoder.contrastive_update(anchor, positive,negative, enc_idx)
            self.decoder.contrastive_update(anchor, positive,negative, dec_idx)

        return enc_vals, enc_idx, recon

class MeaningFormLayer(nn.Module):
    def __init__(self, dim, k, lr=1e-3):
        super().__init__()
        self.meaning = BioBrainLayer(dim, k, lr=lr)
        self.form    = BioBrainLayer(dim, k, lr=lr)
        self.dim = dim
        self.k   = k
        self.lr  = lr

    @torch.no_grad()
    def forward(self, active_vals, active_idx):
        meaning_vals, meaning_idx = self.meaning(active_vals, active_idx)

        form_vals, form_idx = self.form(meaning_vals, meaning_idx)

        return meaning_vals, meaning_idx, form_vals, form_idx

    @torch.no_grad()
    def bind(self, meaning_vals, meaning_idx, form_vals, form_idx, reward=1.0):



        pred_form = self.meaning.predict(meaning_vals, meaning_idx)
        self.meaning.predictive_update(
            x_real=form_vals,
            x_pred=pred_form,
            active_vals=meaning_vals,
            active_idx=meaning_idx
        )

        pred_meaning = self.form.predict(form_vals, form_idx)
        self.form.predictive_update(
            x_real=meaning_vals,
            x_pred=pred_meaning,
            active_vals=form_vals,
            active_idx=form_idx
        )


        anchor = meaning_vals.mean(dim=0)

        positive = form_vals.mean(dim=0)

        negative = torch.randn_like(positive)

        anchor = anchor * reward
        positive = positive * reward

        self.meaning.contrastive_update(anchor, positive, negative)

        self.form.contrastive_update(anchor, positive, negative)
class DiskMemory(nn.Module):
    def __init__(self, dim, base_path="memstore", n_clusters=128, lr=1e-3):
        super().__init__()
        self.dim = dim
        self.lr = lr
        os.makedirs(base_path, exist_ok=True)

        self.p2 = nn.Linear(dim, 8192, bias=False).to(DEVICE).half()
        self.p3 = nn.Linear(8192, 4096, bias=False).to(DEVICE).half()
        self.p4 = nn.Linear(4096, 1024, bias=False).to(DEVICE).half()
        self.p5 = nn.Linear(1024, 256, bias=False).to(DEVICE).half()
        self.back = nn.Linear(256, dim, bias=False).to(DEVICE).half()

        self.wm_size = 64
        self.register_buffer("wm", torch.zeros(self.wm_size, dim, dtype=torch.float16))
        self.wm_ptr = 0

        self.mg1_size = 50000
        self.register_buffer("mg1", torch.zeros(self.mg1_size, 8192, dtype=torch.float16))
        self.mg1_ptr = 0

        self.mg2_size = 50000
        self.register_buffer("mg2", torch.zeros(self.mg2_size, 4096, dtype=torch.float16))
        self.mg2_ptr = 0

        self.ltm1_size = 10_000_000
        self.ltm2_size = 10_000_000

        self.ltm1_path = os.path.join(base_path, "ltm1.bin")
        self.ltm2_path = os.path.join(base_path, "ltm2.bin")

        if not os.path.exists(self.ltm1_path):
            np.memmap(self.ltm1_path, dtype=np.int8, mode="w+", shape=(self.ltm1_size, 1024))
        if not os.path.exists(self.ltm2_path):
            np.memmap(self.ltm2_path, dtype=np.int8, mode="w+", shape=(self.ltm2_size, 256))

        self.ltm1 = np.memmap(self.ltm1_path, dtype=np.int8, mode="r+", shape=(self.ltm1_size, 1024))
        self.ltm2 = np.memmap(self.ltm2_path, dtype=np.int8, mode="r+", shape=(self.ltm2_size, 256))

        self.ltm1_ptr = 0
        self.ltm2_ptr = 0

        self.n_clusters = n_clusters
        self.register_buffer("centroids", torch.randn(n_clusters, 256, device=DEVICE, dtype=torch.float16))
        self.cluster_members = [[] for _ in range(n_clusters)]
        self.centroid_lr = 0.01

        self.faiss_dim = 256
        self.nlist = self.n_clusters
        self.m = 32

        quantizer = faiss.IndexFlatIP(self.faiss_dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            self.faiss_dim,
            self.nlist,
            self.m,
            8
        )
        self.faiss_training_buffer = []
        self.faiss_trained = False

        res = faiss.StandardGpuResources()
        self.index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
        self.index_gpu.nprobe = 4

    @torch.no_grad()
    def _sanitize_vec(self, v):
        return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _quantize_to_int8(self, v):
        v = self._sanitize_vec(v)
        v_cpu = v.detach().cpu().numpy().astype(np.float32)
        return np.clip(v_cpu * 127.0, -128, 127).astype(np.int8)

    @torch.no_grad()
    def _assign_cluster(self, v5_float):
        v5_float = self._sanitize_vec(v5_float)
        sims = F.cosine_similarity(v5_float.unsqueeze(0), self.centroids, dim=1)
        cluster = int(torch.argmax(sims).item())
        self.centroids[cluster] = (1 - self.centroid_lr) * self.centroids[cluster] + self.centroid_lr * v5_float
        return cluster

    @torch.no_grad()
    def store(self, vec):
        vec = self._sanitize_vec(vec.to(DEVICE).half())

        self.wm[self.wm_ptr] = vec
        self.wm_ptr = (self.wm_ptr + 1) % self.wm_size

        v2 = self._sanitize_vec(self.p2(vec))
        self.mg1[self.mg1_ptr] = v2
        self.mg1_ptr = (self.mg1_ptr + 1) % self.mg1_size

        v3 = self._sanitize_vec(self.p3(v2))
        self.mg2[self.mg2_ptr] = v3
        self.mg2_ptr = (self.mg2_ptr + 1) % self.mg2_size

        v4 = self._sanitize_vec(self.p4(v3))
        self.ltm1[self.ltm1_ptr] = self._quantize_to_int8(v4)
        self.ltm1_ptr = (self.ltm1_ptr + 1) % self.ltm1_size

        v5 = self._sanitize_vec(self.p5(v4))
        v5_np = v5.detach().cpu().numpy().astype(np.float32)

        cluster = self._assign_cluster(v5)

        idx = self.ltm2_ptr
        if idx < self.ltm2_size:
            self.ltm2[idx] = self._quantize_to_int8(v5)
            self.ltm2_ptr = (self.ltm2_ptr + 1) % self.ltm2_size
            self.cluster_members[cluster].append(idx)

        if not self.faiss_trained:
            if len(self.faiss_training_buffer) < 10000:
                self.faiss_training_buffer.append(v5_np)
            if len(self.faiss_training_buffer) == 10000:
                train_data = np.stack(self.faiss_training_buffer)
                self.index_gpu.train(train_data)
                self.faiss_trained = True
                del self.faiss_training_buffer

        if self.faiss_trained:
            self.index_gpu.add(v5_np.reshape(1, -1))

    @torch.no_grad()
    def retrieve_topk(self, query, k=16):


        query = query.to(DEVICE).half()
        query = self._sanitize_vec(query)

        q2 = self._sanitize_vec(self.p2(query))
        q3 = self._sanitize_vec(self.p3(q2))
        q4 = self._sanitize_vec(self.p4(q3))
        q5 = self._sanitize_vec(self.p5(q4))

        q_np = q5.detach().cpu().numpy().astype(np.float32)

        distances, indices = self.index_gpu.search(q_np.reshape(1, -1), k)
        indices = indices[0]

        if indices[0] == -1:
            empty = torch.zeros(k, 256, device=DEVICE, dtype=torch.float16)
            return empty, torch.zeros(k, dtype=torch.long, device=DEVICE)

        valid = indices[indices >= 0]
        mem_np = self.ltm2[valid].astype(np.float32) / 127.0
        mem = torch.from_numpy(mem_np).to(DEVICE).half()
        mem = self._sanitize_vec(mem)

        return mem, torch.from_numpy(valid).to(DEVICE).long()

    @torch.no_grad()
    def predictive_update(self, query_vec, retrieved_vec):


        query_vec = torch.nan_to_num(query_vec, nan=0.0)
        retrieved_vec = torch.nan_to_num(retrieved_vec, nan=0.0)

        error = (retrieved_vec - query_vec)

        grad = torch.matmul(error.unsqueeze(1), query_vec.unsqueeze(0))

        grad = self.lr * grad

        MAX_NORM = 0.5
        gnorm = grad.norm()
        if gnorm > MAX_NORM:
            grad.mul_(MAX_NORM / (gnorm + 1e-8))

        self.back.weight.data -= grad.half()


    @torch.no_grad()
    def get_hard_negatives(self, anchor_vec, k=8):
        anchor_vec = anchor_vec.to(DEVICE).half()
        anchor_vec = self._sanitize_vec(anchor_vec)

        a2 = self._sanitize_vec(self.p2(anchor_vec))
        a3 = self._sanitize_vec(self.p3(a2))
        a4 = self._sanitize_vec(self.p4(a3))
        a5 = self._sanitize_vec(self.p5(a4))

        sims = F.cosine_similarity(a5.unsqueeze(0), self.centroids, dim=1)
        pos_cluster = int(torch.argmax(sims).item())

        sorted_clusters = torch.argsort(sims, descending=True).tolist()
        neg_clusters = [c for c in sorted_clusters if c != pos_cluster][:4]

        candidates = []
        for cluster in neg_clusters:
            members = self.cluster_members[cluster]
            if len(members) == 0:
                continue
            take = members[-4096:] if len(members) > 4096 else members
            candidates.extend(take)

        if len(candidates) == 0:
            return torch.randn(k, 256, device=DEVICE).half()

        candidates = np.array(candidates, dtype=np.int64)
        candidates = candidates[candidates < self.ltm2_size]

        mem_np = self.ltm2[candidates].astype(np.float32) / 127.0
        mem = torch.from_numpy(mem_np).to(DEVICE).half()
        mem = self._sanitize_vec(mem)

        sims_mem = F.cosine_similarity(a5.unsqueeze(0), mem, dim=1)
        top_idx = torch.topk(sims_mem, k).indices
        hard_neg = mem[top_idx]

        return hard_neg
    @torch.no_grad()
    def contrastive_update(self, anchor, positive):


        T = 0.07
        MAX_NORM = 0.5

        anchor = anchor.detach().to(DEVICE).half()
        positive = positive.detach().to(DEVICE).half()

        anchor = self._sanitize_vec(anchor)
        positive = self._sanitize_vec(positive)

        a5 = self._sanitize_vec(self.p5(self._sanitize_vec(self.p4(self._sanitize_vec(self.p3(self._sanitize_vec(self.p2(anchor))))))))
        p5 = self._sanitize_vec(self.p5(self._sanitize_vec(self.p4(self._sanitize_vec(self.p3(self._sanitize_vec(self.p2(positive))))))))

        a = F.normalize(a5, dim=-1)
        p = F.normalize(p5, dim=-1)

        hard_negs = self.get_hard_negatives(anchor, k=8)
        hard_negs = self._sanitize_vec(hard_negs)
        n = F.normalize(hard_negs, dim=-1)
        n_mean = n.mean(dim=0)

        sim_pos = (a * p).sum() / T
        sim_neg = (a * n_mean).sum() / T

        error = torch.tanh(sim_pos - sim_neg)

        sims_centroids = F.cosine_similarity(a.unsqueeze(0), self.centroids, dim=1)
        pos_cluster = int(torch.argmax(sims_centroids).item())

        dw_pos = self.lr * error * a
        if dw_pos.norm() > MAX_NORM:
            dw_pos = dw_pos * (MAX_NORM / (dw_pos.norm() + 1e-8))

        self.centroids[pos_cluster] += dw_pos

        sorted_clusters = torch.argsort(sims_centroids, descending=True).tolist()
        neg_clusters = [c for c in sorted_clusters if c != pos_cluster][:4]

        dw_neg = self.lr * error * n_mean
        if dw_neg.norm() > MAX_NORM:
            dw_neg = dw_neg * (MAX_NORM / (dw_neg.norm() + 1e-8))

        for c in neg_clusters:
            self.centroids[c] -= dw_neg

        self.centroids.data = F.normalize(self.centroids.data, dim=-1)

class FullModel(nn.Module):
    def __init__(self, dim, k, n_layers=4, update_interval=8, lr=1e-3):
        super().__init__()

        self.autoenc = AutoencoderLayer(dim, k, lr=lr)
        self.layers  = nn.ModuleList([BioBrainLayer(dim, k, lr=lr) for _ in range(n_layers)])
        self.meaning_form = MeaningFormLayer(dim, k, lr=lr)
        self.memory = DiskMemory(dim, lr=lr)

        self.update_interval = update_interval
        self.step = 0
        self.dim = dim
        self.k = k
        self.lr = lr

    @torch.no_grad()
    def _sanitize_dense_(self, x):
        x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return x

    @torch.no_grad()
    def _safe_active_idx(self, x):
        B, D = x.shape
        row0 = x[0]
        mask = (row0 != 0) & torch.isfinite(row0)
        active_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

        if active_idx.numel() == 0:
            active_idx = torch.arange(D, device=DEVICE)

        return torch.clamp(active_idx.long(), 0, D - 1)

    @torch.no_grad()
    def forward(self, x):
        self.step += 1
        do_update = (self.step % self.update_interval == 0)

        x = x.to(DEVICE).half()
        self._sanitize_dense_(x)

        z, z_idx, recon = self.autoenc(x, do_update=do_update)

        active_vals = z
        active_idx  = z_idx

        for i, layer in enumerate(self.layers):

            active_vals, active_idx = layer(active_vals, active_idx)
            self._sanitize_dense_(active_vals)

            if i == 2:
                dense = torch.zeros(self.dim, device=DEVICE, dtype=torch.float16)
                dense[active_idx] = active_vals.mean(dim=0)

                self.memory.store(dense)

                retrieved, _ = self.memory.retrieve_topk(dense, k=16)
                mem_vec = retrieved.mean(dim=0) if retrieved.numel() > 0 else torch.zeros(256, device=DEVICE)

                mem_proj = self.memory.back(mem_vec)
                fused = dense + mem_proj
                fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

                fused_batch = fused.unsqueeze(0).repeat(active_vals.size(0), 1)
                new_idx = self._safe_active_idx(fused_batch)
                active_vals = fused_batch[:, new_idx]
                active_idx  = new_idx

                if do_update:
                    self.memory.predictive_update(query_vec=dense, retrieved_vec=mem_proj)

                    negative = self.memory.get_hard_negatives(anchor=dense, k=8).mean(dim=0)

                    self.memory.contrastive_update(anchor=dense, positive=mem_proj)

        meaning_vals, meaning_idx, form_vals, form_idx = self.meaning_form(active_vals, active_idx)

        if do_update:
            self.meaning_form.bind(meaning_vals, meaning_idx, form_vals, form_idx)

        return {
            "meaning_vals": meaning_vals,
            "meaning_idx": meaning_idx,
            "form_vals": form_vals,
            "form_idx": form_idx,
            "recon": recon,
        }

class BioConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 lr=1e-3):
        super().__init__()
        self.lr = lr

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
        ).to(DEVICE).half()

    @torch.no_grad()
    def forward(self, x):
        x = x.half().to(DEVICE)
        y = self.conv(x)
        return y

    @torch.no_grad()
    def predictive_update(self, x_real, x_pred, active_vals, reward=1.0):


        error = x_real - x_pred
        error = torch.nan_to_num(error, nan=0.0)

        err_mean = error.mean(dim=(0, 2, 3))

        act_mean = active_vals.mean(dim=(0, 2, 3))

        dw = torch.ger(err_mean, act_mean)
        dw = dw[:, :, None, None]
        dw = clamp_update_(dw)

        self.conv.weight.data += self.lr * reward * dw.half()
        self.conv.bias.data   += self.lr * reward * err_mean.half()

    @torch.no_grad()
    def contrastive_update(self, anchor, positive, negative, reward=1.0):


        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negative, dim=-1)

        sim_pos = (a * p).sum()
        sim_neg = (a * n).sum()

        error = (sim_pos - sim_neg) * reward

        dw = error * a

        dw = dw[:, None, None, None].half()
        dw = clamp_update_(dw)
        self.conv.weight.data += self.lr * dw


class AudioFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.register_buffer(
            "mel_fb",
            torch.randn(n_mels, n_fft // 2 + 1, device=DEVICE).abs().half()
        )

    @torch.no_grad()
    def waveform_to_mel(self, wav):
        wav_f32 = wav.to(DEVICE).float()

        window = torch.hann_window(self.n_fft, device=DEVICE).float()

        spec = torch.stft(
            wav_f32,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        mag = spec.abs()
        mel = torch.matmul(self.mel_fb.float(), mag)
        mel = torch.log(mel + 1e-6)

        mel = mel.unsqueeze(1).permute(0, 1, 3, 2)
        return mel.half()

    @torch.no_grad()
    def vad(self, wav, energy_thresh=0.01):
        wav_f32 = wav.to(DEVICE).float()
        energy = (wav_f32 ** 2).mean(dim=-1)
        return energy > energy_thresh

    @torch.no_grad()
    def forward(self, wav):
        mel = self.waveform_to_mel(wav)
        vad_mask = self.vad(wav)
        return mel, vad_mask


class AudioEncoder(nn.Module):
    def __init__(self, dim=MODEL_DIM, k=K):
        super().__init__()

        self.conv1 = BioConv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv2 = BioConv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = BioConv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = BioConv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.fc = nn.Linear(7168, dim).to(DEVICE).half()
    @torch.no_grad()
    def forward(self, mel, vad_mask, do_update=True, reward=1.0):
        mel = mel.half().to(DEVICE)

        is_event = bool(vad_mask.item())

        y1 = F.relu(self.conv1(mel))
        y2 = F.relu(self.conv2(y1))
        y3 = F.relu(self.conv3(y2))
        y4 = F.relu(self.conv4(y3))

        y4_mean = y4.mean(dim=(2,3))
        y4_max  = y4.amax(dim=(2,3))

        fused = torch.cat([y4_mean, y4_max], dim=1)

        dense = self.fc(fused).half()

        B, D = dense.shape
        active_idx = torch.nonzero(dense[0] != 0, as_tuple=False).squeeze(1)
        if active_idx.numel() == 0:
            active_idx = torch.arange(D, device=DEVICE)
        active_idx = torch.clamp(active_idx.long(), 0, D - 1)

        active_vals = dense[:, active_idx]

        if do_update and is_event:
            self.conv1.predictive_update(x_real=mel, x_pred=y1, active_vals=mel, reward=reward)
            self.conv2.predictive_update(x_real=y1,  x_pred=y2, active_vals=y1, reward=reward)
            self.conv3.predictive_update(x_real=y2,  x_pred=y3, active_vals=y2, reward=reward)
            self.conv4.predictive_update(x_real=y3,  x_pred=y4, active_vals=y3, reward=reward)

            anchor   = y4_mean
            positive = y3.mean(dim=(2,3))
            negative = torch.randn_like(positive)
            self.conv4.contrastive_update(anchor, positive, negative)

        return active_vals, active_idx, is_event




class VisionEncoder(nn.Module):
    def __init__(self, img_size=128, dim=MODEL_DIM, k=K, lr=1e-3):
        super().__init__()

        self.conv1 = BioConv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = BioConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = BioConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = BioConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc = nn.Linear(16384, dim).to(DEVICE).half()

        self.register_buffer("prev_frame",
            torch.zeros(1, 3, img_size, img_size, dtype=torch.float16)
        )

        self.lr = lr
    @torch.no_grad()
    def forward(self, x, do_update=True, reward=1.0):
        x = x.half().to(DEVICE)

        diff = x - self.prev_frame
        diff = torch.nan_to_num(diff, nan=0.0)

        motion_energy = diff.abs().mean().item()
        is_event = motion_energy > 1e-3

        diff = torch.where(diff.abs() < 1e-3, torch.zeros_like(diff), diff)

        self.prev_frame = x.clone()

        y1 = F.relu(self.conv1(diff))
        y2 = F.relu(self.conv2(y1))
        y3 = F.relu(self.conv3(y2))
        y4 = F.relu(self.conv4(y3))

        y4_mean = y4.mean(dim=(2,3))

        y4_max  = y4.amax(dim=(2,3))

        saliency_map = diff.abs().mean(dim=1, keepdim=True)
        saliency_map = F.interpolate(saliency_map, size=y4.shape[2:], mode="bilinear")
        y4_sal = (y4 * saliency_map).mean(dim=(2,3))

        fused = torch.cat([y4_mean, y4_max, y4_sal], dim=1)

        dense = self.fc(fused).half()

        B, D = dense.shape
        active_idx = torch.nonzero(dense[0] != 0, as_tuple=False).squeeze(1)
        if active_idx.numel() == 0:
            active_idx = torch.arange(D, device=DEVICE)
        active_idx = torch.clamp(active_idx.long(), 0, D - 1)

        active_vals = dense[:, active_idx]

        if do_update and is_event:
            self.conv1.predictive_update(x_real=diff, x_pred=y1, active_vals=diff, reward=reward)
            self.conv2.predictive_update(x_real=y1,  x_pred=y2, active_vals=y1, reward=reward)
            self.conv3.predictive_update(x_real=y2,  x_pred=y3, active_vals=y2, reward=reward)
            self.conv4.predictive_update(x_real=y3,  x_pred=y4, active_vals=y3, reward=reward)

            anchor   = y4_mean
            positive = y3.mean(dim=(2,3))
            negative = torch.randn_like(positive)
            self.conv4.contrastive_update(anchor, positive, negative)

        return active_vals, active_idx, is_event


class EmbodiedSpeechModel(nn.Module):
    def __init__(self, dim=MODEL_DIM, k=K):
        super().__init__()

        self.vision = VisionEncoder(img_size=128, dim=dim, k=k).to(DEVICE)
        self.audio_frontend = AudioFrontend().to(DEVICE)
        self.audio_encoder = AudioEncoder(dim=dim, k=k).to(DEVICE)
        self.brain = FullModel(dim, k).to(DEVICE)

        self.fusion_layer = BioBrainLayer(dim, k).to(DEVICE)
    @torch.no_grad()
    def step(self, frame, wav):
        frame = frame.half().to(DEVICE)
        wav   = wav.half().to(DEVICE)

        v_vals, v_idx, v_event = self.vision(frame)

        mel, vad_mask = self.audio_frontend(wav)
        a_vals, a_idx, a_event = self.audio_encoder(mel, vad_mask)

        is_event = v_event or a_event

        B, dim = v_vals.size()
        dense_v = torch.zeros(B, dim, device=DEVICE, dtype=torch.float16)
        dense_a = torch.zeros(B, dim, device=DEVICE, dtype=torch.float16)
        dense_v[:, v_idx] = v_vals
        dense_a[:, a_idx] = a_vals * vad_mask.to(DEVICE).view(B, 1)

        fused_dense = dense_v + dense_a
        fused_dense = torch.nan_to_num(fused_dense.float(), nan=0.0)

        full_idx = torch.arange(dim, device=DEVICE)
        fused_vals, fused_idx = self.fusion_layer(fused_dense, full_idx)

        out = self.brain(fused_vals if is_event else fused_vals.detach(), is_event=is_event)

        return out, vad_mask

if __name__ == "__main__":
    DIM = MODEL_DIM
    K = max(1, int(DIM * SPARSITY))

    model = EmbodiedSpeechModel(DIM, K).to(DEVICE)

    frame = torch.randn(1, 3, 128, 128, device=DEVICE)
    wav = torch.randn(1, 16000, device=DEVICE)

    out, vad = model.step(frame, wav)

    print("VAD (spraak gedetecteerd):", bool(vad.item()))
    print("meaning_vals:", out["meaning_vals"].shape)
    print("form_vals:", out["form_vals"].shape)
    print("recon:", out["recon"].shape)
