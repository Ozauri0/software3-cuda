import csv, math
from collections import defaultdict

def load(path):
    rows=[]
    with open(path,'r') as f:
        for r in csv.DictReader(f): rows.append(r)
    return rows

local = load('results/resultados_procesado.csv')
server = load('results/resultados-server-u.csv')

def avg(rows, inst, var, pop, field='time_total_ms'):
    vals=[float(r[field]) for r in rows if r['instance']==inst and r['variant']==var and r['population']==str(pop) and r.get(field,'')]
    return sum(vals)/len(vals) if vals else None

def sd(rows, inst, var, pop, field='time_total_ms'):
    vals=[float(r[field]) for r in rows if r['instance']==inst and r['variant']==var and r['population']==str(pop) and r.get(field,'')]
    if len(vals)<2: return 0
    m=sum(vals)/len(vals)
    return math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1))

errors = []

def check(label, pres, real, tol=5):
    ok = abs(pres - real) < tol
    status = "OK" if ok else "ERROR"
    if not ok:
        errors.append(f"{label}: pres={pres} real={real:.1f}")
    print(f"  {status} {label}: pres={pres} actual={real:.1f}")

print("=== REVISION COMPLETA ===\n")

# --- SLIDE 5: RTX 4090 ---
print("--- SLIDE 5: RTX 4090 ---")
pres_4090 = {
    ('small',1024): (387, 65),
    ('small',4096): (1626, 73),
    ('small',16384): (6813, 101),
    ('medium',1024): (3497, 309),
    ('medium',4096): (15186, 326),
    ('medium',16384): (62574, 388),
    ('large',1024): (19846, 1474),
    ('large',4096): (79994, 1994),
    ('large',16384): (328125, 4684),
}
for (inst,pop) in pres_4090:
    p_cpu, p_opt = pres_4090[(inst,pop)]
    r_cpu = avg(server, inst, 'cpu', pop)
    r_opt = avg(server, inst, 'cuda_opt', pop)
    check(f"4090 {inst} pop={pop} CPU", p_cpu, r_cpu)
    check(f"4090 {inst} pop={pop} Opt", p_opt, r_opt)

# --- SLIDE 6: RTX 3060 ---
print("\n--- SLIDE 6: RTX 3060 ---")
pres_3060 = {
    ('small',1024): (606, 106),
    ('small',4096): (2686, 121),
    ('small',16384): (11447, 398),
    ('medium',1024): (7798, 531),
    ('medium',4096): (33094, 668),
    ('medium',16384): (132015, 2980),
    ('large',1024): (27569, 2327),
    ('large',4096): (113597, 3390),
    ('large',16384): (518126, 14262),
}
for (inst,pop) in pres_3060:
    p_cpu, p_opt = pres_3060[(inst,pop)]
    r_cpu = avg(local, inst, 'cpu', pop)
    r_opt = avg(local, inst, 'cuda_opt', pop)
    check(f"3060 {inst} pop={pop} CPU", p_cpu, r_cpu)
    check(f"3060 {inst} pop={pop} Opt", p_opt, r_opt)

# --- SLIDE 7: Comparativa ---
print("\n--- SLIDE 7: Comparativa ---")
comp = {
    ('small',4096): (73, 121),
    ('medium',4096): (326, 668),
    ('medium',16384): (388, 2980),
    ('large',4096): (1994, 3390),
    ('large',16384): (4684, 14262),
}
for (inst,pop) in comp:
    p4090, p3060 = comp[(inst,pop)]
    r4090 = avg(server, inst, 'cuda_opt', pop)
    r3060 = avg(local, inst, 'cuda_opt', pop)
    check(f"Comp {inst} pop={pop} 4090", p4090, r4090)
    check(f"Comp {inst} pop={pop} 3060", p3060, r3060)

# --- SLIDE 7: Ratios ---
print("\n--- SLIDE 7: Ratios ---")
# Pop 4096
ratios = []
for inst in ['small','medium','large']:
    r4090 = avg(server, inst, 'cuda_opt', 4096)
    r3060 = avg(local, inst, 'cuda_opt', 4096)
    ratios.append(r3060/r4090)
avg_r = sum(ratios)/len(ratios)
check("Ratio pop=4096", 1.7, avg_r, 0.3)

# Pop 16384
ratios = []
for inst in ['small','medium','large']:
    r4090 = avg(server, inst, 'cuda_opt', 16384)
    r3060 = avg(local, inst, 'cuda_opt', 16384)
    ratios.append(r3060/r4090)
avg_r = sum(ratios)/len(ratios)
check("Ratio pop=16384", 7.7, avg_r, 0.5)

# CPU ratio
cpu_r = avg(local,'medium','cpu',4096) / avg(server,'medium','cpu',4096)
check("CPU ratio medium", 2.2, cpu_r, 0.2)

# --- SLIDE 8: Chart ---
print("\n--- SLIDE 8: Chart ---")
chart = {
    ('4090',1024): 11.3, ('4090',4096): 46.6, ('4090',16384): 161.1,
    ('3060',1024): 14.7, ('3060',4096): 49.6, ('3060',16384): 44.3,
}
for (gpu,pop) in chart:
    if gpu == '4090':
        c = avg(server,'medium','cpu',pop)
        o = avg(server,'medium','cuda_opt',pop)
    else:
        c = avg(local,'medium','cpu',pop)
        o = avg(local,'medium','cuda_opt',pop)
    actual = c/o
    check(f"Chart {gpu} pop={pop}", chart[(gpu,pop)], actual, 0.5)

# --- SLIDE 9: Quality ---
print("\n--- SLIDE 9: Quality ---")
pres_q = {
    'small': (3147, 3166, 3166),
    'medium': (136235, 136073, 136073),
    'large': (1609922, 1631447, 1631447),
}
for inst in ['small','medium','large']:
    for i,var in enumerate(['cpu','cuda','cuda_opt']):
        vals=[float(r['best_value']) for r in local if r['instance']==inst and r['variant']==var]
        r = sum(vals)/len(vals) if vals else 0
        p = pres_q[inst][i]
        check(f"Quality {inst} {var}", p, r, 100)

# --- SLIDE 9: Factibility ---
print("\n--- SLIDE 9: Factibility ---")
pres_f = {'small':(55.8,55.8,55.8), 'medium':(36.3,36.3,36.3), 'large':(0.4,0.4,0.4)}
for inst in ['small','medium','large']:
    for i,var in enumerate(['cpu','cuda','cuda_opt']):
        vals=[float(r['feasible_pct']) for r in local if r['instance']==inst and r['variant']==var]
        r = sum(vals)/len(vals) if vals else 0
        p = pres_f[inst][i]
        check(f"Factibility {inst} {var}", p, r, 0.2)

# --- SLIDE 9: Block size ---
print("\n--- SLIDE 9: Block size ---")
for bs,pres in [(128,645),(256,669),(512,905)]:
    vals=[float(r['time_total_ms']) for r in local if r['variant']==f'cuda_opt_b{bs}']
    r = sum(vals)/len(vals) if vals else 0
    check(f"Block {bs}", pres, r, 2)

# --- SLIDE 9: Transfers ---
print("\n--- SLIDE 9: Transfers ---")
for var,h2d,d2h in [('cuda',1.8,3.4),('cuda_opt',1.8,3.3)]:
    vals_h=[float(r['time_h2d_ms']) for r in local if r['instance']=='medium' and r['variant']==var and r['population']=='4096']
    vals_d=[float(r['time_d2h_ms']) for r in local if r['instance']=='medium' and r['variant']==var and r['population']=='4096']
    r_h = sum(vals_h)/len(vals_h) if vals_h else 0
    r_d = sum(vals_d)/len(vals_d) if vals_d else 0
    check(f"Transfer {var} H2D", h2d, r_h, 0.5)
    check(f"Transfer {var} D2H", d2h, r_d, 0.5)

# --- SLIDE 9: Transfer % ---
print("\n--- SLIDE 9: Transfer % ---")
vals_h=[float(r['time_h2d_ms']) for r in local if r['instance']=='medium' and r['variant']=='cuda_opt' and r['population']=='4096']
vals_d=[float(r['time_d2h_ms']) for r in local if r['instance']=='medium' and r['variant']=='cuda_opt' and r['population']=='4096']
vals_t=[float(r['time_total_ms']) for r in local if r['instance']=='medium' and r['variant']=='cuda_opt' and r['population']=='4096']
pct = (sum(vals_h)/len(vals_h) + sum(vals_d)/len(vals_d)) / (sum(vals_t)/len(vals_t)) * 100
check("Transfer %", 0.7, pct, 0.3)

# --- TITLE ---
print("\n--- TITLE ---")
r_med_opt = avg(server,'medium','cuda_opt',16384)
r_med_cpu = avg(server,'medium','cpu',16384)
max_spd = r_med_cpu/r_med_opt
check("Max speedup", 161, max_spd, 2)

n_local = len(local)
n_server = len(server)
total = n_local + n_server
check("Total experiments", 570, total, 5)

# --- SUMMARY ---
print(f"\n=== RESUMEN ===")
print(f"Errores encontrados: {len(errors)}")
if errors:
    for e in errors:
        print(f"  - {e}")
else:
    print("  Todos los datos coinciden correctamente!")
