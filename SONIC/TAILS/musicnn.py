# import logging
# from tqdm.auto import tqdm
# from CREAM import dataset
# from TAILS import embedder
# from musicnn.extractor import extractor
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization

# class MusicNNEmbedder(embedder.Embedder):
#     def __init__(self, batch_size=32):
#         super().__init__(batch_size)
#         logging.info("Initialized MusicNNEmbedder")
#         model = Sequential([
#             Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#             BatchNormalization(),
#             Flatten(),
#             Dense(128, activation='relu'),
#             BatchNormalization(),
#             Dense(10, activation='softmax')
#         ])
#     def embedding_fn(self, waveform):
#         """
#         Compute MusicNN embeddings for the given waveform.
#         Parameters:
#             waveform (torch.Tensor): Input waveform.
#         Returns:
#             list: List of embeddings extracted by MusicNN.
#         """
#         # Assuming the waveform is a valid audio tensor, convert it to a numpy array
#         # since musicnn expects a path or numpy array, not a tensor.
#         waveform_np = waveform.cpu().numpy()
#         tags, embeddings = extractor(waveform_np, model='MSD_musicnn')
#         return embeddings

#     def get_embeddings(self, audio_dir):
#         """
#         Get embeddings for all audio files in the given directory.
#         Parameters:
#             audio_dir (str): Path to the directory containing audio files.
#         Returns:
#             list: List of tuples (audio_path, embedding).
#         """
#         dataloader = dataset.init_dataset(audio_dir, batch_size=self.batch_size, transform=self.embedding_fn)
#         embeddings = []
        
#         logging.info("Computing MusicNN embeddings")
#         logging.info(f"Using device: {self.device}")

#         for i, batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
#             logging.info(f"Processing batch {i + 1}/{len(dataloader)}")
#             for audio_path, embedding in zip(*batch):
#                 embeddings.append((audio_path, embedding))
#                 logging.debug(f"Computed MusicNN embedding for {audio_path}")

#         return embeddings

# # Example model definition using BatchNormalization
