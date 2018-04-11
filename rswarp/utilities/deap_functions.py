import random

"""
Using `random` package for random number generation. 
This is to keep in line with the implementation in the DEAP library.
Note that `random.randint` has inclusive lower and upper bounds 
(differs from `np.random.randint`). 
"""

tec_template = {
    'T_coll': {'type': 'float',
               'min': 50.0,
               'max': 900.0,
               'mutation': {
                   'probability': 0.25,
                   'r': 0.1,
                   'k': 10.0
               }},
    'T_em': {'type': 'float',
             'min': 900.0,
             'max': 1900.0,
             'mutation': {
                 'probability': 0.25,
                 'r': 0.1,
                 'k': 10.0
             }},
    'V_grid': {'type': 'float',
               'min': 0.0,
               'max': 20.0,
               'mutation': {
                   'probability': 0.25,
                   'r': 0.1,
                   'k': 10.0
               }},
    'gap_distance': {'type': 'float',
                     'min': 1e-6,
                     'max': 20e-6,
                     'mutation': {
                         'probability': 0.25,
                         'r': 0.1,
                         'k': 10.0
                     }},
    'grid_height': {'type': 'float',
                    'min': 0.05,
                    'max': 0.95,
                    'mutation': {
                        'probability': 0.25,
                        'r': 0.1,
                        'k': 10.0
                    }},
    'phi_coll': {'type': 'float',
                 'min': 1.5,
                 'max': 4.0,
                 'mutation': {
                     'probability': 0.25,
                     'r': 0.1,
                     'k': 10.0
                 }},
    'phi_em': {'type': 'float',
               'min': 1.5,
               'max': 4.0,
               'mutation': {
                   'probability': 0.25,
                   'r': 0.1,
                   'k': 10.0
               }},
    'rho_cw': {'type': 'float',
               'min': 1e-5,
               'max': 1e-2,
               'mutation': {
                   'probability': 0.25,
                   'r': 0.1,
                   'k': 10.0
               }},
    'rho_ew': {'type': 'float',
               'min': 1e-5,
               'max': 1e-2,
               'mutation': {
                   'probability': 0.25,
                   'r': 1,
                   'k': 10.0
               }},
    'rho_load': {'type': 'float',
                 'min': 1e-4,
                 'max': 0.1,
                 'mutation': {
                     'probability': 0.25,
                     'r': 0.1,
                     'k': 10.0
                 }},
    'strut_height': {'type': 'float',
                     'min': 10e-9,
                     'max': 20e-9,
                     'mutation': {
                         'probability': 0.25,
                         'r': 0.1,
                         'k': 10.0
                     }},
    'strut_width': {'type': 'float',
                    'min': 10e-9,
                    'max': 20e-9,
                    'mutation': {
                        'probability': 0.25,
                        'r': 0.1,
                        'k': 10.0
                    }},
    'x_struts': {'type': 'int',
                 'min': 1,
                 'max': 4,
                 'mutation': {
                     'probability': 0.25,
                     'r': 0.5,
                     'k': 2
                 }},
    'y_struts': {'type': 'int',
                 'min': 1,
                 'max': 4,
                 'mutation': {
                     'probability': 0.25,
                     'r': 0.5,
                     'k': 2
                 }}
}


def initDict(container, func):
    return func(container())


def generate_new_tec(adict, template=tec_template):
    for key, value in template.iteritems():
        value_type = template[key]['type']
        minimum, maximum = template[key]['min'], template[key]['max']

        if value_type == 'int':
            gen_num = random.randint(minimum, maximum)
        elif value_type == 'float':
            gen_num = minimum + random.random() * (maximum - minimum)
        else:
            raise TypeError("Type must be int or float")
        adict[key] = gen_num

    return adict


def mutIntorGauss(individual, template=tec_template):
    for key in individual:
        if template[key]['type'] == 'int':
            indpbInteger = template[key]['mutation']['probability']
            if random.random() < indpbInteger:
                xl, xu = template[key]['mutation']['lower'], template[key]['mutation']['upper']
                individual[key] = random.randint(xl, xu)
        if template[key]['type'] == 'float':
            indpbGauss = template[key]['mutation']['probability']
            if random.random() < indpbGauss:
                m, s = template[key]['mutation']['mean'], template[key]['mutation']['std']
                if random.random() < 0.5:
                    sgn = -1.0
                else:
                    sgn = +1.0
                individual[key] = abs(individual[key] + sgn * random.gauss(m, s))
    return individual,


def mutBoundedExp(individual, template=tec_template):
    """
    Describes a variable step size bounded mutation operator.
    Performs Var_i = Var_i + s_i * r_i * a_i. Chance for variable to mutate should be given in template.
    s_i: random from {-1, 1}
    r_i: r * domain_i

    If the resulting Var_i is not within the variables bounds it will be set to it's closest bound.
    From:
    Muhlenbein, H. and Schlierkamp-Voosen, D.:
    Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization. Evolutionary Computation,
    1 (1), pp. 25-49, 1993.
    Args:
        individual: Individual created from DEAP
        r: Range factor, normally in [0.1, 1e-6]
        k: mutation precision, normally in {4, 5, ... 20}
        template: Dictionary providing template for each variables constraints and properties.

    Returns:

    """
    for key in individual:
        r = template[key]['mutation']['r']
        k = template[key]['mutation']['k']
        indpbInteger = template[key]['mutation']['probability']
        if random.random() >= indpbInteger:
            continue

        si = random.sample([-1.0, 1.0], 1)[0]
        ri = r * random.uniform(template[key]['min'], template[key]['max'])
        u = random.random()
        ai = 2**(-u * k)

        individual[key] = individual[key] + si * ri * ai
        if template[key]['type'] == int:
            individual[key] = int(individual[key])
        if individual[key] < template[key]['min']:
            individual[key] = template[key]['min']
        elif individual[key] > template[key]['max']:
            individual[key] = template[key]['max']

    return individual,


def cxUniform(ind1, ind2, prob=0.5, template=tec_template):
    # TODO: What should prob be set to? Probability for any given gene to crossover if two individuals do cross over
    """
    Execute a gene level cross over with probability of prob for any given gene mutation.
    """
    k1, k2, r1, r2 = [random.uniform(0, 1) for _ in range(3)]
    for key in ind1:
        if random.random() < prob:
            S1, S2 = (1. + r1) * (k1 * ind1[key] + (1. - k1) * ind2[key]), \
                     (1. + r2) * (k2 * ind1[key] + (1. - k2) * ind2[key])
            if S1 < template[key]['min'] or S1 > template[key]['max']:
                S1 = S1 / (1. + r1)
            if S2 < template[key]['min'] or S2 > template[key]['max']:
                S2 = S2 / (1. + r1)
            if template[key]['type'] == 'int':
                ind1[key], ind2[key] = int(S1), int(S2)
            else:
                ind1[key], ind2[key] = int(S1), int(S2)

    return ind1, ind2
