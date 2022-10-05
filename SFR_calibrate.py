import basic


def run(model_name, ml = None, cond = None):

    info, swr_info, sfr_info, riv_keys_info = basic.load_params(model_name)

    if ml is None:
        ml = basic.load_model()

    sf = ml.sfr

    rch = sf.reach_data

    if cond is None:
        cond = info['parameters']['SFR']['hcond1']

    rch['strhc1'] = cond

    sf.reach_data = rch

    sf.write_file()

    return sf