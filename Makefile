install:
	cd /content/tu-eind-AGSMCTS/playground &&\
		python3 -m pip install --upgrade pip &&\
		pip install -U . &&\
		cd /content/tu-eind-AGSMCTS/cython-env &&\
		mkdir installation &&\
		python3 setup.py develop -d /content/tu-eind-AGSMCTS/cython-env/installation

train:
	cd /content/tu-eind-AGSMCTS/src &&\
		python3 learning_test.py
  