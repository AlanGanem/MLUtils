class Binner:
    '''
    Fits data frame into selected intervals and transforms (bins)
    other dataframes into those bins
    '''

    def __init__(self):
        return

    def fit(self, df, bins):
        assert bins.__class__ == dict
        self.mins = {}
        self.maxs = {}
        self.bins = {i: sorted(bins[i]) for i in bins}
        return self

    def transform(self, df, cross_column_mapper=None):
        df = df.copy()
        bins = self.bins

        if not cross_column_mapper.__class__ == dict:
            ccm = {i: i for i in bins}
        else:
            ccm = {**{i: i for i in bins}, **cross_column_mapper}

        for bn in ccm:
            for i in range(len(bins[ccm[bn]]) + 1):
                # left bound
                if i == 0:
                    df.loc[df[bn] < self.bins[ccm[bn]][0], 'bin_' + bn] = str(
                        '<{}'.format(self.bins[ccm[bn]][0]))

                    # df.loc[df[bn].between(self.mins[ccm[bn]], bins[ccm[bn]][0]), 'bin_' + bn] = str(
                    #    '{} to {}'.format(self.mins[ccm[bn]], bins[ccm[bn]][0]))
                # general case
                elif i == max(range(len(bins[ccm[bn]]) + 1)):
                    # right bound
                    df.loc[df[bn] > self.bins[ccm[bn]][-1], 'bin_' + bn] = str(
                        '>{}'.format(self.bins[ccm[bn]][-1]))
                else:
                    df.loc[df[bn].between(bins[ccm[bn]][i - 1], bins[ccm[bn]][i]), 'bin_' + bn] = str(
                        '{} to {}'.format(bins[ccm[bn]][i - 1], bins[ccm[bn]][i]))

        return df

    def fit_transform(self, df, bins):
        self.fit(df, bins)
        return self.transform(df)