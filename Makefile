install:
	cd ~/tu-eind-AGSMCTS/playground &&\
		python3 -m pip install --upgrade pip &&\
		pip install -U . &&\
		cd ~/tu-eind-AGSMCTS/cython-env &&\
		mkdir installation &&\
		python3 setup.py develop 
		

train:
	cd ~/tu-eind-AGSMCTS/src &&\
		python3 learning_test.py
  
