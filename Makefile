install:
	cd $PWD/tu-eind-AGSMCTS/playground &&\
		pip install -U . &&\
		cd $PWD/tu-eind-AGSMCTS/cython-env &&\
		touch installation &&\
		python3 setup.py develop -d $PWD/tu-eind-AGSMCTS/cython-env/installation

train:
	cd $PWD/tu-eind-AGSMCTS/src &&\
		python3 learning_test.py
  