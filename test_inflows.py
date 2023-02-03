import unittest

# __import__("write_inflows.py")
import write_inflows
# 'write_inflows.py')
import basic

class TestSum(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        # data = [1, 2, 3]
        # result = sum(data)
        for datestart in ['1/1/2012', '1/1/2013']:
            numdays = 365
            out_folder = None

            minvalue = 29.54
            max_value = 38,
            cleandamdata = False
            m = basic.load_model()

            rr = write_inflows.load_riv(station='11464000', title='Russian River', file='RRinflow.dat', figurename='russian_river.png',
                          datestart=datestart, out_folder=out_folder, m=m, numdays=numdays, save_fig=False,
                          write_output=False)

            dry = write_inflows.load_riv(station='11465350', title='Dry Creek', file='Dry_creek.dat', figurename='dry_creek.png',
                           datestart=datestart, out_folder=out_folder, m=m, numdays=numdays, save_fig=False,
                           write_output=False)

            mw = write_inflows.load_riv(station='11466800', title='Mark West Creek', file='MarkWest.dat', figurename='mark_west.png',
                          datestart=datestart, out_folder=out_folder, m=m, numdays=numdays, save_fig=False,
                          write_output=False)

            total = dry.loc[:, 'Q'] + rr.loc[:, 'Q']
            total = total.to_frame('rrtotal')

            stg = write_inflows.load_dam(total, datestart=datestart, minvalue=minvalue, max_value=max_value, numdays=numdays,
                           clean=cleandamdata)

            # write_inflows.plot_dam(stg,minvalue,max_value,out_folder,save_fig=True)

            self.assertEqual(stg.loc[:,'Value'].isnull().sum(), 0)

if __name__ == '__main__':
    unittest.main()
