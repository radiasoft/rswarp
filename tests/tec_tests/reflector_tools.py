import numpy as np

def get_reflector_emission(reflector_data, reflected_js, start_time, end_time):
    # If end time is not none then returns current ->  return = [C/s]
    # If end time == None then returns total charge emitted from start_time until sim end  -> return = [C]
    reflector_lookup = {
        'time': 0,
        'dt': 1,
        'jsid': 2,
        'charge_specular': 3,
        'charge_diffuse': 4
    }

    if not hasattr(reflected_js, '__iter__'):
        reflected_js = [reflected_js, ]

    diffuse_reflected_current, specular_reflected_current = 0., 0.
    for js in reflected_js:
        indices = np.where((reflector_data[:, reflector_lookup['jsid']] == js))[0]

        if not end_time:
            indices = np.where((reflector_data[:, reflector_lookup['time']] > start_time) &
                               (reflector_data[:, reflector_lookup['jsid']] == js))[0]
            # end_time = np.max(reflector_data[:, reflector_lookup['time']])
        else:
            indices = np.where((reflector_data[:, reflector_lookup['time']] > start_time) &
                               (reflector_data[:, reflector_lookup['time']] < end_time) &
                               (reflector_data[:, reflector_lookup['jsid']] == js))[0]

        specular_reflected_current += np.sum(reflector_data[indices, reflector_lookup['charge_specular']])
        diffuse_reflected_current += np.sum(reflector_data[indices, reflector_lookup['charge_diffuse']])

    if end_time:
        return (specular_reflected_current + diffuse_reflected_current) / (end_time - start_time)
    else:
        return (specular_reflected_current + diffuse_reflected_current)

def get_species_current(scraper_data, species_jsid, start_time, end_time):
    # If end time is not none then returns current ->  return = [C/s]
    # If end time == None then returns total charge emitted from start_time until sim end  -> return = [C]

    # Takes scraper_data from rswarp.particlecollector.particlereflector.analyze_collected_charge
    #   (assumes list ordered by species.js)
    # scraper_data is a list of `lostparticles_data` for all species
    # locally define lostparticles_data indices for a helpful reminder
    scraper_lookup = {
        'time': 0,
        'charge': 1,  # Warp records macroparticle_weight * number_macroparticles * macrop_charge == physical charge
        'dt': 2,
        'jsid': 3
    }

    total_current = 0.
    for species_data in scraper_data:
        if species_data.size == 0 or np.any(species_data[:, scraper_lookup['jsid']].astype(int) != species_jsid):
            # This species isn't here or is not the designated species
            continue
        if not end_time:
            indices = np.where((species_data[:, scraper_lookup['time']] > start_time))[0]
            # end_time = np.max(species_data[:, scraper_lookup['time']])
        else:
            indices = np.where((species_data[:, scraper_lookup['time']] > start_time) &
                               (species_data[:, scraper_lookup['time']] < end_time))[0]

        total_species_charge_collected = np.sum(species_data[indices, scraper_lookup['charge']])

        total_current += total_species_charge_collected

    if end_time:
        return total_current / (end_time - start_time)
    else:
        return total_current


def get_total_current(scraper_data, start_time, end_time, area):
    # scraper_data is a list of `lostparticles_data` for each species
    # locally define lostparticles_data indices for a helpful reminder
    scraper_lookup = {
        'time': 0,
        'charge': 1,  # Warp records macroparticle_weight * number_macroparticles == physical charge
        'dt': 2,
        'jsid': 3
    }

    total_current = 0.
    for species_id in range(len(scraper_data)):
        total_current += get_species_current(scraper_data, species_id, start_time, end_time)

    print(total_current, total_current / area)
    return total_current