import subprocess
import os

def training(model_size, batch_size):
    try:
        subprocess.run(
            ["python3", "train_model.py", 
             "--model-size", model_size,
             "--batch-size", str(batch_size)],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
      return False

def exists_dir_iters(model_size):
    # Definisci il percorso della cartella del modello 
    # (Modificalo se lo script salva i dati altrove, es. f"runs/{model_size}")
    model_dir = f"model_{model_size}" 
    
    if not os.path.isdir(model_dir):
        return False
    
    # Prendiamo tutti i file/cartelle dentro la directory del modello
    content = os.listdir(model_dir)
    
    # Filtriamo e cerchiamo se esistono cartelle che iniziano con "iter_"
    iter_dirs = [d for d in content if d.startswith("iter_") and os.path.isdir(os.path.join(model_dir, d))]
    
    # Ritorna True se ha trovato almeno una cartella iter_ (es. iter_0)
    return len(iter_dirs) > 0

# consider that GPU T4 is adopted, for 3070 we could adopt different values
models_batches = [
  ("n",64),
  ("s",32),
  ("m",16),
  ("l",8),
  ("x",4)
]

for m,b in models_batches:
  if exists_dir_iters(m):
    continue
  if training(m,b) is True:
    print(f"Model {m} trained successfully")