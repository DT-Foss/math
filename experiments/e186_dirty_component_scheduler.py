from wirkung_math.dirty import GeometryCostModel, Interval, schedule_components_optimal
from _common import emit

model=GeometryCostModel.from_two_component_gap_fit(
    merged_intercept_ms=5.822425785524378,
    merged_ms_per_gap=0.02049972202198457,
    separate_ms=9.257104014977813,
)
rows=[]
for gap in [0,4,8,16,32,64,128,256,384]:
    left=Interval(0,32)
    right=Interval(32+gap,64+gap)
    plan=schedule_components_optimal([left,right],model)
    rows.append({
        "gap":gap,
        "recommended":"merged" if len(plan)==1 else "tiled",
        "predicted_cost_ms":model.partition_cost(plan),
    })
emit({
    "experiment":"E186",
    "model":{
        "launch_ms":model.launch_ms,
        "cell_ms":model.cell_ms,
        "crossover_gap":model.crossover_gap,
        "reported_fit_r2":0.9596333477606858,
    },
    "rows":rows,
})
