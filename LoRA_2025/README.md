

## Environment configuration

1. Ensure that **CUDA 12.1** and corresponding drivers are installed on your system.
2. python=3.10.16
3. Installation of dependencies：
```bash
pip install -r requirements.txt
cd CLIP-main
python setup.py install  # Install custom CLIP components
cd ..
rm -rf CLIP-main
```
4. Upload your huggingface token in configs.py
## run
```bash
bash run.sh
```