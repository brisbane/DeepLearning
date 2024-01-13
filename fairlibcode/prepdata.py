from fairlib import datasets
#####hyperparameters
bert_batch_size=512
#######################
#datasets.prepare_dataset("bios", "./data/bios")
bios = datasets.bios.Bios(dest_folder="./datatest", batch_size=bert_batch_size)
bios.download_files()
