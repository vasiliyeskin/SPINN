If you have problem <<If you have problem No GPU/TPU found, falling back to CPU>> it can be resolv by installation cuda in conda:

        conda install cuda -c nvidia



Best path to installation is

        pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        
        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        pip install flax
        pip install tqdm
        pip install matplotlib

or

        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html