import basic



def run(name, ml = None, cond = None):
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(name)

    if cond is None:
        cond = info['parameters']['GHB']['cond']

    if ml is None:
        ml = basic.load_model()

    ghb = ml.ghb

    sp = ghb.stress_period_data[0]

    sp['cond'] = cond

    ghb.stress_period_data = sp

    print(f're-writing ghb file with {cond}ft/s cond')

    ghb.write_file()

    return ghb
