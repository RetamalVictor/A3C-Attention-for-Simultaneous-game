install:
	cd ~/A3C-Attention-for-Simultaneous-game/playground &&\
		python3 -m pip install --upgrade pip &&\
		pip install -U . &&\
		cd ~/tu-eind-AGSMCTS/cython-env &&\
		python3 setup.py develop 
		

train:
	cd ~/tu-eind-AGSMCTS/src &&\
		python3 learn.py
  
