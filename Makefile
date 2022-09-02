install:
	python3 -m venv .pommerman &&\
		source .pommerman/bin/activate &&\
		python3 -m pip install --upgrade pip &&\
		pip install -r requirements.txt &&\
		cd ~/A3C-Attention-for-Simultaneous-game/playground &&\
		pip install -U . &&\
		cd ~/A3C-Attention-for-Simultaneous-game/cython-env &&\
		python3 setup.py develop -d ~/.pommerman/lib/python3.8/site-packages
		
train:
	cd ~/tu-eind-AGSMCTS/src &&\
		python3 learn.py