import expyriment

exp = expyriment.design.Experiment(name="First Experiment")
expyriment.control.initialize(exp)

expyriment.control.start()

expyriment.control.end()