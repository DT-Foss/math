from wirkung_math.plan_graph import RepresentationPlanGraph

def test_consumer_changes_optimal_path():
    g=RepresentationPlanGraph()
    g.add("edit","cpu_dense_k",6.46,"materialize CPU K")
    g.add("edit","gpu_dense_k",5.56,"V100 full K")
    g.add("edit","cpu_rank_delta",1.0,"build row/column rank delta")
    g.add("cpu_dense_k","full_k",0.0,"return dense K")
    g.add("gpu_dense_k","full_k",0.0,"return dense K")
    g.add("cpu_rank_delta","cpu_solve",1.1,"Woodbury solve")
    g.add("gpu_dense_k","gpu_solve",28.9,"GPU solve")
    g.add("cpu_solve","solve",0.0,"return solve")
    g.add("gpu_solve","solve",0.0,"return solve")
    assert g.shortest("edit","full_k").states[1]=="gpu_dense_k"
    assert g.shortest("edit","solve").states[1]=="cpu_rank_delta"
