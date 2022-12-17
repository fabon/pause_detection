from torch.utils.data import DataLoader
from data import arap_symbols, PauseDataset
from train import train
from evaluate import evaluate
from config import EPOCHS,LR,BATCH_SIZE,EMBEDDING_SIZE, NUM_CLASS, DEVICE

def main():
    model = PauseClassificationModel(VOCAB_SIZE, EMBEDDING_SIZE, NUM_CLASS).to(DEVICE)

    from torch.utils.data.dataset import random_split
    from torchtext.data.functional import to_map_style_dataset
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    
    total_accu = None

    # TODO: implement train/eval/test splits here
    dataset = PauseDataset("dataset/train.txt", "dataset")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            collate_fn=dataset.collate_fn)

    # TODO implement tensor format in dataloader: context window left right centered around candidate phoneme + binary label, check tensor formats)
    
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(dataloader)
        accu_val = evaluate(dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        # TODO: give results on train/eval splits here
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)

        # TODO: give results on test here
        
if __name__ == "__main__":
    main()
