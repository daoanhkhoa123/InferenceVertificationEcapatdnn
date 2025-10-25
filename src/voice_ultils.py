import io
import ffmpeg
import soundfile as sf
import numpy as np
import torch, torchaudio
import torch.nn.functional as F
from fastapi import UploadFile

def load_parameters(self, path, device = None):
    self_state = self.state_dict()
    loaded_state = torch.load(path, map_location=device)
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "").replace("speaker_encoder.", "")
            if name not in self_state:
                print("%s is not in the model."%origname)
                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)

def _load_any_format(file, sr: int = 16000):
    file.file.seek(0)
    data = file.file.read()

    try:
        file.file.seek(0)
        audio, sr = sf.read(file.file, dtype='float32')
        return audio, sr
    except Exception:
        import ffmpeg
        out, _ = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar=sr)
            .run(input=data, capture_stdout=True, capture_stderr=True, quiet=True)
        )
        audio, sr = sf.read(io.BytesIO(out), dtype='float32')
        return audio, sr

def get_embedding(model, file:UploadFile, device):
    audio, sr= _load_any_format(file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = torch.FloatTensor(audio).to(device).unsqueeze(0)

    with torch.no_grad():
        emb = model(audio, False) 
    return F.normalize(emb, p=2, dim=1)  

def cosine_score(emb1, emb2):
    return torch.mean(F.cosine_similarity(emb1, emb2)).item()
