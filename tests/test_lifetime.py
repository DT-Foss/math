from wirkung_math.lifetime import ExecutionRegime,HorizonPlanner

def test_horizon_route():
    p=HorizonPlanner((
        ExecutionRegime("gpu_full",0,12.8),
        ExecutionRegime("cpu_eager",64.9,2.3),
        ExecutionRegime("cpu_local",0,4.0),
    ))
    assert p.choose(1).name=="cpu_local"
    assert p.choose(100).name=="cpu_eager"
