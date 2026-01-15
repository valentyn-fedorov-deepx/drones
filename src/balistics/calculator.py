import math
from typing import List, Tuple

# Constants
GRAVITY = -32.174
M_PI = math.pi
__BCOMP_MAXRANGE__ = 50000

# Drag function constants
G0 = 0
G1 = 1
G2 = 2
G5 = 5
G6 = 6
G7 = 7
G8 = 8

# Angular conversion functions
def deg_to_moa(deg: float) -> float:
    return deg * 60

def deg_to_rad(deg: float) -> float:
    return deg * M_PI / 180

def moa_to_deg(moa: float) -> float:
    return moa / 60

def moa_to_rad(moa: float) -> float:
    return moa / 60 * M_PI / 180

def rad_to_deg(rad: float) -> float:
    return rad * 180 / M_PI

def rad_to_moa(rad: float) -> float:
    return rad * 60 * 180 / M_PI

# Atmospheric correction functions
def calc_fr(temperature: float, pressure: float, relative_humidity: float) -> float:
    vpw = 4e-6 * temperature**3 - 0.0004 * temperature**2 + 0.0234 * temperature - 0.2517
    frh = 0.995 * (pressure / (pressure - 0.3783 * relative_humidity * vpw))
    return frh

def calc_fp(pressure: float) -> float:
    pstd = 29.53  # in-hg
    return (pressure - pstd) / pstd

def calc_ft(temperature: float, altitude: float) -> float:
    tstd = -0.0036 * altitude + 59
    return (temperature - tstd) / (459.6 + tstd)

def calc_fa(altitude: float) -> float:
    fa = -4e-15 * altitude**3 + 4e-10 * altitude**2 - 3e-5 * altitude + 1
    return 1 / fa

def atm_correct(drag_coefficient: float, altitude: float, barometer: float, 
                temperature: float, relative_humidity: float) -> float:
    fa = calc_fa(altitude)
    ft = calc_ft(temperature, altitude)
    fr = calc_fr(temperature, barometer, relative_humidity)
    fp = calc_fp(barometer)
    # Calculate the atmospheric correction factor
    cd = fa * (1 + ft - fp) * fr
    return drag_coefficient * cd

def retard(drag_function: int, drag_coefficient: float, velocity: float) -> float:
    """
    Calculate drag retardation based on the given drag function and velocity.
    """
    vp = velocity
    val = -1
    a = -1
    m = -1
    
    # Handle different drag functions
    if drag_function == G0:
        if vp > 2600:
            a = 1.5366e-03
            m = 1.67
        elif vp > 2000:
            a = 5.8497e-03
            m = 1.5
        elif vp > 1460:
            a = 5.9814e-04
            m = 1.8
        elif vp > 1190:
            a = 9.5408e-08
            m = 3.0
        elif vp > 1040:
            a = 2.3385e-18
            m = 6.45
        elif vp > 840:
            a = 5.9939e-08
            m = 3.0
        elif vp > 0:
            a = 7.4422e-04
            m = 1.6

    elif drag_function == G1:
        if vp > 4230:
            a = 1.477404177730177e-04
            m = 1.9565
        elif vp > 3680:
            a = 1.920339268755614e-04
            m = 1.925
        elif vp > 3450:
            a = 2.894751026819746e-04
            m = 1.875
        elif vp > 3295:
            a = 4.349905111115636e-04
            m = 1.825
        elif vp > 3130:
            a = 6.520421871892662e-04
            m = 1.775
        elif vp > 2960:
            a = 9.748073694078696e-04
            m = 1.725
        elif vp > 2830:
            a = 1.453721560187286e-03
            m = 1.675
        elif vp > 2680:
            a = 2.162887202930376e-03
            m = 1.625
        elif vp > 2460:
            a = 3.209559783129881e-03
            m = 1.575
        elif vp > 2225:
            a = 3.904368218691249e-03
            m = 1.55
        elif vp > 2015:
            a = 3.222942271262336e-03
            m = 1.575
        elif vp > 1890:
            a = 2.203329542297809e-03
            m = 1.625
        elif vp > 1810:
            a = 1.511001028891904e-03
            m = 1.675
        elif vp > 1730:
            a = 8.609957592468259e-04
            m = 1.75
        elif vp > 1595:
            a = 4.086146797305117e-04
            m = 1.85
        elif vp > 1520:
            a = 1.954473210037398e-04
            m = 1.95
        elif vp > 1420:
            a = 5.431896266462351e-05
            m = 2.125
        elif vp > 1360:
            a = 8.847742581674416e-06
            m = 2.375
        elif vp > 1315:
            a = 1.456922328720298e-06
            m = 2.625
        elif vp > 1280:
            a = 2.419485191895565e-07
            m = 2.875
        elif vp > 1220:
            a = 1.657956321067612e-08
            m = 3.25
        elif vp > 1185:
            a = 4.745469537157371e-10
            m = 3.75
        elif vp > 1150:
            a = 1.379746590025088e-11
            m = 4.25
        elif vp > 1100:
            a = 4.070157961147882e-13
            m = 4.75
        elif vp > 1060:
            a = 2.938236954847331e-14
            m = 5.125
        elif vp > 1025:
            a = 1.228597370774746e-14
            m = 5.25
        elif vp > 980:
            a = 2.916938264100495e-14
            m = 5.125
        elif vp > 945:
            a = 3.855099424807451e-13
            m = 4.75
        elif vp > 905:
            a = 1.185097045689854e-11
            m = 4.25
        elif vp > 860:
            a = 3.566129470974951e-10
            m = 3.75
        elif vp > 810:
            a = 1.045513263966272e-08
            m = 3.25
        elif vp > 780:
            a = 1.291159200846216e-07
            m = 2.875
        elif vp > 750:
            a = 6.824429329105383e-07
            m = 2.625
        elif vp > 700:
            a = 3.569169672385163e-06
            m = 2.375
        elif vp > 640:
            a = 1.839015095899579e-05
            m = 2.125
        elif vp > 600:
            a = 5.71117468873424e-05
            m = 1.950
        elif vp > 550:
            a = 9.226557091973427e-05
            m = 1.875
        elif vp > 250:
            a = 9.337991957131389e-05
            m = 1.875
        elif vp > 100:
            a = 7.225247327590413e-05
            m = 1.925
        elif vp > 65:
            a = 5.792684957074546e-05
            m = 1.975
        elif vp > 0:
            a = 5.206214107320588e-05
            m = 2.000

    elif drag_function == G2:
        if vp > 1674:
            a = .0079470052136733
            m = 1.36999902851493
        elif vp > 1172:
            a = 1.00419763721974e-03
            m = 1.65392237010294
        elif vp > 1060:
            a = 7.15571228255369e-23
            m = 7.91913562392361
        elif vp > 949:
            a = 1.39589807205091e-10
            m = 3.81439537623717
        elif vp > 670:
            a = 2.34364342818625e-04
            m = 1.71869536324748
        elif vp > 335:
            a = 1.77962438921838e-04
            m = 1.76877550388679
        elif vp > 0:
            a = 5.18033561289704e-05
            m = 1.98160270524632

    elif drag_function == G5:
        if vp > 1730:
            a = 7.24854775171929e-03
            m = 1.41538574492812
        elif vp > 1228:
            a = 3.50563361516117e-05
            m = 2.13077307854948
        elif vp > 1116:
            a = 1.84029481181151e-13
            m = 4.81927320350395
        elif vp > 1004:
            a = 1.34713064017409e-22
            m = 7.8100555281422
        elif vp > 837:
            a = 1.03965974081168e-07
            m = 2.84204791809926
        elif vp > 335:
            a = 1.09301593869823e-04
            m = 1.81096361579504
        elif vp > 0:
            a = 3.51963178524273e-05
            m = 2.00477856801111

    elif drag_function == G6:
        if vp > 3236:
            a = 0.0455384883480781
            m = 1.15997674041274
        elif vp > 2065:
            a = 7.167261849653769e-02
            m = 1.10704436538885
        elif vp > 1311:
            a = 1.66676386084348e-03
            m = 1.60085100195952
        elif vp > 1144:
            a = 1.01482730119215e-07
            m = 2.9569674731838
        elif vp > 1004:
            a = 4.31542773103552e-18
            m = 6.34106317069757
        elif vp > 670:
            a = 2.04835650496866e-05
            m = 2.11688446325998
        elif vp > 0:
            a = 7.50912466084823e-05
            m = 1.92031057847052

    elif drag_function == G7:
        if vp > 4200:
            a = 1.29081656775919e-09
            m = 3.24121295355962
        elif vp > 3000:
            a = 0.0171422231434847
            m = 1.27907168025204
        elif vp > 1470:
            a = 2.33355948302505e-03
            m = 1.52693913274526
        elif vp > 1260:
            a = 7.97592111627665e-04
            m = 1.67688974440324
        elif vp > 1110:
            a = 5.71086414289273e-12
            m = 4.3212826264889
        elif vp > 960:
            a = 3.02865108244904e-17
            m = 5.99074203776707
        elif vp > 670:
            a = 7.52285155782535e-06
            m = 2.1738019851075
        elif vp > 540:
            a = 1.31766281225189e-05
            m = 2.08774690257991
        elif vp > 0:
            a = 1.34504843776525e-05
            m = 2.08702306738884

    elif drag_function == G8:
        if vp > 3571:
            a = .0112263766252305
            m = 1.33207346655961
        elif vp > 1841:
            a = .0167252613732636
            m = 1.28662041261785
        elif vp > 1120:
            a = 2.20172456619625e-03
            m = 1.55636358091189
        elif vp > 1088:
            a = 2.0538037167098e-16
            m = 5.80410776994789
        elif vp > 976:
            a = 5.92182174254121e-12
            m = 4.29275576134191
        elif vp > 0:
            a = 4.3917343795117e-05
            m = 1.99978116283334

    # Calculate and return the final value if we have valid coefficients
    if a != -1 and m != -1 and 0 < vp < 10000:
        val = a * pow(vp, m) / drag_coefficient
        return val
    
    return -1

# Solution retrieval functions
def get_range(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage] if yardage < size else 0

def get_path(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 1] if yardage < size else 0

def get_moa(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 2] if yardage < size else 0

def get_time(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 3] if yardage < size else 0

def get_windage(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 4] if yardage < size else 0

def get_windage_moa(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 5] if yardage < size else 0

def get_velocity(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 6] if yardage < size else 0

def get_vx(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 7] if yardage < size else 0

def get_vy(sln: List[float], yardage: int) -> float:
    size = sln[__BCOMP_MAXRANGE__ * 10 + 1]
    return sln[10 * yardage + 8] if yardage < size else 0

def windage(wind_speed: float, vi: float, xx: float, t: float) -> float:
    vw = wind_speed * 17.60  # Convert to inches per second
    return vw * (t - xx / vi)

def head_wind(wind_speed: float, wind_angle: float) -> float:
    wangle = deg_to_rad(wind_angle)
    return math.cos(wangle) * wind_speed

def cross_wind(wind_speed: float, wind_angle: float) -> float:
    wangle = deg_to_rad(wind_angle)
    return math.sin(wangle) * wind_speed

def zero_angle(drag_function: int, drag_coefficient: float, vi: float, 
               sight_height: float, zero_range: float, y_intercept: float) -> float:
    t = 0
    dt = 1 / vi
    y = -sight_height / 12
    x = 0
    da = deg_to_rad(14)  # Initial angular change
    quit = False
    
    for angle in range(0, int(deg_to_rad(45) * 10000), 1):
        angle = angle / 10000  # Convert back to radians
        vy = vi * math.sin(angle)
        vx = vi * math.cos(angle)
        gx = GRAVITY * math.sin(angle)
        gy = GRAVITY * math.cos(angle)
        
        t, x, y = 0, 0, -sight_height / 12
        
        while x <= zero_range * 3:
            vy1, vx1 = vy, vx
            v = math.sqrt(vx**2 + vy**2)
            dt = 1 / v
            dv = retard(drag_function, drag_coefficient, v)
            dvy = -dv * vy / v * dt
            dvx = -dv * vx / v * dt
            vx += dvx
            vy += dvy
            vy += dt * gy
            vx += dt * gx
            x += dt * (vx + vx1) / 2
            y += dt * (vy + vy1) / 2
            
            if vy < 0 and y < y_intercept:
                break
                
            if vy > 3 * vx:
                break
            
            t += dt
            
        if y > y_intercept and da > 0:
            da = -da / 2
        elif y < y_intercept and da < 0:
            da = -da / 2
            
        if abs(da) < moa_to_rad(0.01):
            quit = True
            break
            
    return rad_to_deg(angle)

def solve_all(drag_function: int, drag_coefficient: float, vi: float, 
              sight_height: float, shooting_angle: float, z_angle: float,
              wind_speed: float, wind_angle: float) -> Tuple[List[float], int]:
    dt = 0.5 / vi
    headwind = head_wind(wind_speed, wind_angle)
    crosswind = cross_wind(wind_speed, wind_angle)
    gy = GRAVITY * math.cos(deg_to_rad(shooting_angle + z_angle))
    gx = GRAVITY * math.sin(deg_to_rad(shooting_angle + z_angle))
    
    solution = []
    n = 0
    vx = vi * math.cos(deg_to_rad(z_angle))
    vy = vi * math.sin(deg_to_rad(z_angle))
    y = -sight_height / 12
    x = 0
    t = 0
    
    while True:
        vx1, vy1 = vx, vy
        v = math.sqrt(vx**2 + vy**2)
        dt = 0.5 / v
        
        dv = retard(drag_function, drag_coefficient, v + headwind)
        dvx = -(vx / v) * dv
        dvy = -(vy / v) * dv
        
        vx += dt * dvx + dt * gx
        vy += dt * dvy + dt * gy
        
        if x / 3 >= n:
            solution.extend([
                x / 3,                    # Range in yards
                y * 12,                   # Path in inches
                -rad_to_moa(math.atan(y / x if x != 0 else float('inf'))),  # Correction in MOA
                t + dt,                   # Time in seconds
                windage(crosswind, vi, x, t + dt),  # Windage in inches
                rad_to_moa(math.atan(solution[-1] / 12 / (x / 3)) if x != 0 else 0),  # Windage in MOA
                v,                        # Velocity (combined)
                vx,                       # Velocity (x)
                vy,                       # Velocity (y)
                0                         # Reserved
            ])
            n += 1

        x += dt * (vx + vx1) / 2
        y += dt * (vy + vy1) / 2
        t += dt

        if abs(vy) > abs(3 * vx) or n >= __BCOMP_MAXRANGE__ + 1:
            break

    return solution, n
