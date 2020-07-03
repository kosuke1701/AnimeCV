import torch

def metric_l2(tensor1, tensor2):
    return - torch.norm(tensor1 - tensor2, dim=1)

class Similarity(object):
    def __init__(self, sim_func, batch_size=100000):
        if isinstance(sim_func, str):
            if sim_func == "L2":
                self.sim_func = metric_l2
        else:
            self.sim_func = sim_func
        self.bs = batch_size
    
    def compute_similarity(self, tensor1, tensor2, mode):
        if mode == "pair":
            d = tensor1.size(1)
            n1 = tensor1.size(0)
            n2 = tensor2.size(0)

            # Split larger tensor to reduce peak memory usage.
            n_chunk = max(n1 * n2 // self.bs, 1)
            if n1 >= n2:
                chunks = torch.chunk(tensor1, n_chunk, dim=0)
                sims = []
                for chunk in chunks:
                    _n1 = chunk.size(0)
                    sims.append(
                        self.sim_func(
                            chunk.repeat(1, n2).view(_n1*n2, d),
                            tensor2.repeat(_n1, 1)
                        ).view(_n1, n2)
                    )
                if len(sims) > 1:
                    sims = torch.cat(sims, dim=0)
                else:
                    sims = sims[0]
            else:
                chunks = torch.chunk(tensor2, n_chunk, dim=0)
                sims = []
                for chunk in chunks:
                    _n2 = chunk.size(0)
                    sims.append(
                        self.sim_func(
                            tensor1.repeat(1, _n2).view(n1*_n2, d),
                            chunk.repeat(n1, 1)
                        ).view(n1, _n2)
                    )
                if len(sims) > 1:
                    sims = torch.cat(sims, dim=1)
                else:
                    sims = sims[0]
            
            return sims

        elif mode == "batch":
            return self.sim_func(tensor1, tensor2)
        

