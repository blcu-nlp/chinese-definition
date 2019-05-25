import torch
import torch.utils.data as data
import numpy as np


class DefseqDataset(data.Dataset):
    def __init__(self, root, mode='train'):
        """Set the path for dataset.
        Args:
            root: data directory.
        """
        if mode not in ['train', 'valid', 'test']:
            raise ValueError("mode must be train/valid/test")
        self.mode = mode
        data = np.load(root)
        self.words = data['words']
        self.sememes = data['sememes']
        if self.mode != 'test':
            self.definitions = data['definitions']

    def __getitem__(self, index):
        """Returns one data pair ( word, sememe, definition )."""
        word = torch.LongTensor(self.words[index])
        sememes = torch.LongTensor(self.sememes[index])
        if self.mode != 'test':
            definition = torch.LongTensor(self.definitions[index])
            item = (word, sememes, definition)
        else:
            item = (word, sememes)
        return item

    def __len__(self):
        return len(self.words)


def get_loader(root, batch_size, shuffle=True, num_workers=2, mode='train'):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    defseq = DefseqDataset(root=root, mode=mode)

    data_loader = data.DataLoader(
        dataset=defseq,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return data_loader
