from farmer_augmented import *

ws = w

cp = ws.add_checkpoint()

mi = cp.add_modelinstance()

model.run(checkpoint=cp)

crop = mi.sync_db.add_set("crop", 1, "crop type")

ph_W = mi.sync_db.add_parameter_dc("ph_W", [crop,], "ph weight")
xbar = mi.sync_db.add_parameter_dc("xbar", [crop,], "ph xbar")

W_on = mi.sync_db.add_parameter("W_on", 0, "activate w term")
prox_on = mi.sync_db.add_parameter("prox_on", 0, "activate prox term")

mi.instantiate("simple min negprofit using nlp",
    [
        gams.GamsModifier(ph_W),
        gams.GamsModifier(xbar),
        gams.GamsModifier(W_on),
        gams.GamsModifier(prox_on),
    ],
)

prox_on.add_record().value = 1.0
W_on.add_record().value = 1.0

crop_ext = model.out_db.get_set("crop")
for c in crop_ext:
    name = c.key(0)
    ph_W.add_record(name).value = 50
    xbar.add_record(name).value = 100

mi.solve(output=sys.stdout)

prox_on.find_record().value = 1.0
W_on.find_record().value = 1.0

crop_ext = model.out_db.get_set("crop")
for c in crop_ext:
    name = c.key(0)
    ph_W.find_record(name).value = 500
    xbar.find_record(name).value = 1000

mi.solve(output=sys.stdout)

prox_on.find_record().value = 0.0
W_on.find_record().value = 0.0

mi.solve(output=sys.stdout)
