praca inżynierska <br/>
https://docs.google.com/document/d/1vwCZgo1lLyNIXctKjpwBRIgVVJ1DtH0ubrmZTCOeKFY/edit?usp=sharing <br />
dataset <br/>
https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj
<br/>
Wypakuj tak żeby było:<br/>
data/Celeb-DF-v2/........

To co trzeba zainstalowac jest w requirements.txt, chyba wszystko

Jak już wszystko ustawisz to odpal 
preprocess.py

to ci bierze klatki pojedyncze z filmu i twarze z nich wycina, z 20 min zajmie

trenowanie jeszcze nie jest zrobione


zeby uzywac gpu do trenowania i wgl wszystkiego pobierz(warto, zwłaszcza do preprocessingu): https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local <br/>
zainstaluj<br/>
Potem pobierz <br/>
https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10
<br/>i zainstaluj
potem w folderze projektu:<br/>
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
