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
        self.bins = bins
        for bn in bins:
            self.maxs[bn] = df[bn].max()
            self.mins[bn] = df[bn].min()

    def transform(self, df, cross_column_mapper=None):
        df = df.copy()
        bins = self.bins

        if not cross_column_mapper.__class__ == dict:
            ccm = {i: i for i in bins}
        else:
            ccm = cross_column_mapper

        for bn in ccm:
            for i in range(len(bins[ccm[bn]])):
                if i == 0:
                    df.loc[df[bn].between(self.mins[ccm[bn]], bins[ccm[bn]][0]), 'bin_' + bn] = str(
                        '{} to {}'.format(self.mins[ccm[bn]], bins[ccm[bn]][0]))
                else:
                    df.loc[df[bn].between(bins[ccm[bn]][i - 1], bins[ccm[bn]][i]), 'bin_' + bn] = str(
                        '{} to {}'.format(bins[ccm[bn]][i - 1], bins[ccm[bn]][i]))
                df.loc[df[bn].between(bins[ccm[bn]][i], self.maxs[ccm[bn]]), 'bin_' + bn] = str(
                    '{} to {}'.format(bins[ccm[bn]][i], self.maxs[ccm[bn]]))
        return df

    def fit_transform(self, df, bins):
        self.fit(df, bins)
        return self.transform(df)