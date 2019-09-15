import torch
import torch.multiprocessing as mp


class Population:
    def __init__(self):
        self.swarms = []

    def add(self, swarm) -> None:
        self.swarms.append(swarm)

    def run(self,
            data: torch.Tensor,
            episodes: int,
            ) -> None:
        # cpus = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1

        processes = []
        for swarm in self.swarms:
            processes.append(mp.Process(target=swarm.train, args=(data, episodes)))

        for process in processes:
            process.start()

        for process in processes:
            process.join()
