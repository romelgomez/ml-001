# TODO
# from generator import load_csm_1b
# import torchaudio
# import torch


# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
    
# generator = load_csm_1b(device=device)

# audio = generator.generate(
#     text="Hello from Sesame.",
#     speaker=0,
#     context=[],
#     max_audio_length_ms=10_000,
# )

# torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

