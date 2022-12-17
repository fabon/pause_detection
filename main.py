from torch.utils.data import DataLoader
from data import arap_symbols, PauseDataset
from train import train
from evaluate import evaluate
from config import EPOCHS,LR,BATCH_SIZE,EMBEDDING_SIZE, NUM_CLASS, DEVICE

def main():
    train_iter = AG_NEWS(split='train')
    model = PauseClassificationModel(VOCAB_SIZE, EMBEDDING_SIZE, NUM_CLASS).to(DEVICE)

    from torch.utils.data.dataset import random_split
    from torchtext.data.functional import to_map_style_dataset
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    
    total_accu = None
    train_iter, test_iter = AG_NEWS()
    
    # train_dataset = to_map_style_dataset(train_iter)
    # test_dataset = to_map_style_dataset(test_iter)
    
    # num_train = int(len(train_dataset) * 0.95)
    # split_train_, split_valid_ = \
        
    # random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    # train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
    #                               shuffle=True, collate_fn=collate_batch)
    # valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
    #                               shuffle=True, collate_fn=collate_batch)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
    #                              shuffle=True, collate_fn=collate_batch)


    dataset = PauseDataset("dataset/train.txt", "dataset")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            collate_fn=dataset.collate_fn)

    # for sample in dataloader:
    #     print(sample.phonemes.shape)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(dataloader)
        accu_val = evaluate(dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)

if __name__ == "__main__":
    main()
