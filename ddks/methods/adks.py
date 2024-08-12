import torch
import numpy as np
import warnings
from scipy.special import binom
import logging
from itertools import product

# TODO: Validate the output

def S_(x, f):
    return np.power(-1, np.floor(4.0 * f * x))

def in_Z(x):
    return (int(x) == x) and (x >= 0)


class smooth_max(object):
    def __init__(self, T=0.1):
        self.T = T

    def __call__(self, x):
        return self.T * torch.log((1.0 / len(x))
                                  * torch.sum(torch.exp(x / self.T)))

class adKS(object):
    def __init__(
            self,
            soft=False,
            T=0.1,
            method='all',
            n_test_points=10,
            pts=None,
            norm=False,
            oneway=False
            ):
        
        # Code hygiene statement; self.softge/self.hardge are functions
        if soft:
            self.max = smooth_max(T=T)
            self.ge = self.softge
        else:
            self.max = torch.max
            self.ge = self.hardge

        self.method = method
        self.n_test_points = n_test_points
        self.pts = pts
        self.norm = norm
        self.oneway = oneway

    def __call__(self, pred, true, analytic_distribution, support_lim):
        '''
        Takes in two distributions and returns ddks distance
        For child classes define setup/calcD
        :param pred: [N1 x d] tensor
        :param true: [N2 x d] tensor
        :param analytic_distribution: scipy.stats distribution of the true analytic cdf
        :return:
        '''
        self.pred = torch.Tensor(pred)
        self.true = torch.Tensor(true)
        self.dist = analytic_distribution
        self.support_lim = support_lim

        #Enforce N x d and d1=d2
        if len(pred.shape) < 2 or len(true.shape) < 2:
            warnings.warn(f'Error Pred or True is missing a dimension')
        if pred.shape[1] != true.shape[1]:
            warnings.warn(f'Dimension Mismatch between pred/true: Shapes should be [n1,d], [n2,d]')

        self.setup(self.pred,self.true)

        D = self.calcD(self.pred)

        return D

    def setup(self,pred,true):
        self.getQU(pred,true)

    def M(self, sample, test_points):
        get_ort = self.get_orthants
        _M = get_ort(sample, test_points)
        return _M

    def calcD(self, pred):
        get_ort = self.get_orthants

        # Get orthants
        os_pp = get_ort(pred, self.Q)

        p_coords = self.cdf_coordinates(self.Q.tolist(), self.support_lim)
        os_pt = [self.sort_orthants(self.get_orthant_density(x)) for x in p_coords]

        os_pt = torch.stack(os_pt)

        D1 = self.max((os_pp - os_pt).abs())
        if self.oneway:
            D = D1
        else:
            os_tp = get_ort(pred, self.U)
            t_coords = self.cdf_coordinates(self.U.tolist(), self.support_lim)
            os_tt = torch.stack([self.sort_orthants(self.get_orthant_density(x)) for x in t_coords])
            D2 = self.max((os_tt - os_tp).abs())
            D = max(D1,D2)
        if self.norm:
            D = D / float(pred.shape[0])

        return D
    
    ###
    # Setup Functions
    ###
    def getQU(self, pred, true):
        # Uses self.method to choose to use all points to split space or subsample
        # or use grid
        if self.method == 'all':
            Q = pred;
            U = true
        elif self.method == 'subsample':
            idx = np.random.choice(np.arange(np.min([pred.shape[0], true.shape[0]])), size=self.n_test_points)
            Q = pred[idx, ...];
            U = true[idx, ...]
        elif self.method == 'linear':
            if self.pts is None:
                Q = torch.empty((self.n_test_points, pred.shape[1]))
                U = torch.empty((self.n_test_points, true.shape[1]))
                for dim in range(pred.shape[1]):
                    Q[:, dim] = torch.linspace(pred[:, dim].min(), pred[:, dim].max(), self.n_test_points)
                for dim in range(true.shape[1]):
                    U[:, dim] = torch.linspace(true[:, dim].min(), true[:, dim].max(), self.n_test_points)
        self.Q = Q
        self.U = U
        return
    
    ###
    # calcD functions
    ###

    ###
    # Empirical CDF
    ###
    def get_orthants(self, x, points):
        '''
        n-Dimensional version of get_octants (probably faster to run 3-D using get_orthants)
        :param x: Either pred/true used the samples being placed into orthants
        :param points: The points being used to create orthants i.e Q/U if self.method='all' Q/U = pred/true
        :return: row-Normalized occupancy matrix - each element corresponds to the occupancy % in an orthant
        '''
        N = x.shape[0]
        d = points.shape[1]
        # shape our input and test points into the right shape (N, 3, 1)
        x = x.unsqueeze(-1)
        points = points.unsqueeze(-1)
        # repeat each input point in the dataset across the third dimension
        x = x.repeat((1, 1, points.shape[0]))
        # repeate each test in the dataset across the first dimension
        comp_x = points.repeat((1, 1, x.shape[0]))
        comp_x = comp_x.permute((2, 1, 0))
        # now compare the input points and comparison points to see how many
        # are bigger and smaller
        x = self.ge(x, comp_x)
        orthants = []
        orthant_matrix = self.get_orthant_matrix(d)
        for i in range(int(np.power(2, d))):
            membership = 1.0
            for j in range(d):
                membership *= (float(orthant_matrix[i, j] < 0) + orthant_matrix[i, j] * x[:, j, :]).abs()
            orthant = torch.sum(membership, dim=0).float() / N
            orthants.append(orthant)
        return torch.stack(orthants, dim=1)

    def get_orthant_matrix(self, d):
        n_orthants = int(np.power(2, d))
        x = np.empty((n_orthants, d))
        for i in range(n_orthants):
            for j in range(d):
                x[i, j] = S_(i, np.power(2.0, -j - 2))
        return x
    
    ###
    # Analytic CDF
    ###

    def generate_combinations(self, list1, list2):
        # Combine the two lists into a list of tuples
        combined = list(zip(list1, list2))
        
        # Use itertools.product to get all combinations
        result = list(product(*combined))
        
        return result

    def cdf_coordinates(self, points, max_value):
        
        d = len(points[0])
        N = len(points)

        max_value = [[max_value]*d]*N

        return list(map(self.generate_combinations, points, max_value))
        
    def compare_m_terms(self, m_term, m_terms):
        
        out_terms = [[0] * len(m_term)]

        max_count = m_term.tolist().count(1)


        for term in m_terms:
            for i, j in enumerate(zip(term, m_term)):
                j, k = j
                try:
                    if max_count ==  term.tolist().count(1):
                        break
                except:
                    pass

                if (j != k) and (j == 1):
                    break

                if i == len(term) - 1:
                    out_terms.append(term.tolist())

        return out_terms

    def get_orthant_density(self, coordinates):    

        d = len(coordinates[0])
        max_value = self.support_lim

        n_orthants = int(np.power(2, d))

        densities = self.dist.cdf(coordinates).ravel()
        densities_dict = {point: density for point, density in zip(coordinates, densities)}
        
        orthant_densities = []

        m_count = 0
        scoped_points = list(filter(lambda x: x.count(max_value) == m_count, coordinates))
        scoped_densities = [((np.array(point) == max_value).astype(int), density) for point, density in densities_dict.items() if point in scoped_points]

        while len(orthant_densities) < n_orthants:

            while scoped_densities:
                m_term, current_density = scoped_densities.pop()

                if m_count == 0:
                    orthant_densities.append((m_term, current_density))
                else:
                    m_terms = np.array([m[0] for m in orthant_densities])
                    relevant_terms = self.compare_m_terms(m_term, m_terms)
                    relevant_densities = [density for m, density in orthant_densities if m.tolist() in relevant_terms]
                    orthant_density = current_density - (sum(relevant_densities))
                    orthant_densities.append((m_term, orthant_density))

            m_count += 1
            scoped_points = list(filter(lambda x: x.count(max_value) == m_count, coordinates))
            scoped_densities = [((np.array(point) == max_value).astype(int), density)  for point, density in densities_dict.items() if point in scoped_points]

        return orthant_densities
    
    def sort_orthants(self, densities):
        d = len(densities[0][0])
        orthants = [i[0].tolist() for i in densities]
        orthants.sort(reverse=True)
        sorted_orthants = []
        for orth in orthants:
            orth = orth[::-1]
            density = [e[1] for e in densities if e[0].tolist() == orth][0]
            sorted_orthants.append(density)

        if d==1:
            return torch.Tensor(np.array(sorted_orthants)).squeeze(0)
        else:
            return torch.Tensor(np.array(sorted_orthants))
        
    ###
    #Testing/Validation Functions
    ###
    def p_bi(self, n, m, lam):
        if isinstance(n, float):
            n = np.array([n])
        if isinstance(m, float):
            m = np.array([m])
        _p_bi = binom(m, n) * np.power(lam, n) * np.power(1.0 - lam, m - n)
        _p_bi[np.logical_not(np.isfinite(_p_bi))] = 0.0
        return _p_bi

    def get_n1_n2(self,delta, m_1, m_2):
        #n_1 = np.arange(0, m_1 * (delta + 1) + 1)
        #n_2 = m_2 * (delta + n_1/m_1)
        #_n_2_2 = m_2 * (n_1/m_1 - delta)
        #n_1 = np.concatenate((n_1, n_1))
        #n_2 = np.concatenate((n_2, _n_2_2))
        #idx = np.logical_and(n_1 == n_1.astype(int), n_2 == n_2.astype(int))
        #idx = np.logical_and(idx, n_2 <= m_2)
        #idx = np.logical_and(idx, n_1 <= m_1)
        #idx = np.logical_and(idx, n_2 >= 0)
        #idx = np.logical_and(idx, n_1 >= 0)
        #n_1 = n_1[idx]
        #n_2 = n_2[idx]
        r_1 = np.arange(0.0, m_1 + 0.5) / m_1
        r_2 = np.arange(0.0, m_2 + 0.5) / m_2
        X, Y = np.meshgrid(r_1, r_2)
        x = np.abs(X - Y)
        idx = np.argwhere(np.abs(x - delta) < 1E-6)
        n_1s = m_1 * r_1[idx[:, 1]]
        n_2s = m_2 * r_2[idx[:, 0]]
        return n_1s, n_2s

    def p_delta(self, delta, m_1, m_2, lam):
        _p_delta = 0.0
        n_1, n_2 = self.get_n1_n2(delta, m_1, m_2)
        _p_delta = np.sum(self.p_bi(n_1, m_1, lam) * self.p_bi(n_2, m_2, lam))
        return _p_delta
    
    def p_gtdelta(self, delta, m_1, m_2, lam):
        p_ltdelta = 0.0
        m = max([m_1, m_2])
        d_stars = np.arange(0.0, delta+1/m, 1/m)
        for d_star in d_stars:
            p_ltdelta += self.p_delta(d_star, m_1, m_2, lam)
        return 1.0 - p_ltdelta

    def m_line(self, delta, m_1, m_2):
        return max([m_1, m_2 * (1.0 - delta)])

    def p_D(self, pred=None, true=None, analytic_distribution = None, support_lim = None):
        if pred is None:
            pred = self.pred
        if true is None:
            true = self.true
        if analytic_distribution is None:
            analytic_distribution = self.dist
        if support_lim is None:
            support_lim = self.support_lim

        m_1 = pred.shape[0]
        m_2 = true.shape[0]
        d = true.shape[1]
        D = self(pred, true, analytic_distribution, support_lim).item()
        # round D to the nearest increment by the largest of the sample sizes
        m = m_1*m_2#max([m_1, m_2])
        D = np.round(m*D) / m
        lambda_ik = self.M(true, torch.cat((pred, true))).numpy()
        # _p_D is the probability that every entry in M is less than or equal to
        # D
        _p_D = 1.0
        for i in range(lambda_ik.shape[0]):
            for k in range(lambda_ik.shape[1]):
                p_gtdelta = self.p_gtdelta(D, m_1, m_2, lambda_ik[i, k])
                _p_D *= 1.0 - p_gtdelta

        # we desire to know the probability that something will be larger than D
        return 1.0 - _p_D

    def p(self, pred=None, true=None):
        return self.p_D(pred=pred, true=true)

    def delta_pm(self, delta, m_1, m_2, n_1):
        delta_m = m_2 * ((n_1 / m_1) - delta)
        delta_p = m_2 * ((n_1 / m_1) + delta)
        return delta_p, delta_m

    def permute(self, pred=None, true=None, J=1_000):
        if pred is None:
            pred = self.pred
        if true is None:
            true = self.true
        all_pts = torch.cat((pred, true), dim=0)
        T = self(pred, true)
        T_ = torch.empty((J,))
        total_shape = pred.shape[0] + true.shape[0]
        for j in range(J):
            idx = torch.randperm(total_shape)
            idx1, idx2 = torch.chunk(idx, 2)
            _pred = all_pts[idx1]
            _true = all_pts[idx2]
            T_[j] = self(_pred, _true)
        return float(torch.sum(T < T_) + 1) / float(J + 2), T, T_
    
    ###
    # Utility Functions
    ###
    def softge(self, x, y):
        return (torch.tanh(10.0 * (x - y)) + 1.0) / 2.0

    def hardge(self, x, y):
        return torch.ge(x, y).long()
