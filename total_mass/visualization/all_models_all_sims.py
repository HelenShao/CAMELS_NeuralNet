fig = plt.figure(figsize=(28, 40))
fig.tight_layout()
plt.rcParams['axes.linewidth'] = 1.5

plt.subplot(431).set(xticks=[])
# Symbolic Model

############################### DATA #############################
large_true = np.load("new_model_l2_CAMELS_true.npy")
large_pred = np.load("new_model_l2_CAMELS_pred.npy")

med_true = np.load("new_model_m_CAMELS_true.npy")
med_pred = np.load("new_model_m_CAMELS_pred.npy")

small_true = np.load("new_model_s3_CAMELS_true.npy")
small_pred = np.load("new_model_s3_CAMELS_pred.npy")

total_true = np.concatenate((small_true, med_true, large_true))
total_pred = np.concatenate((small_pred, med_pred, large_pred))

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((total_true - total_pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(total_true)):
    num_total += 1
    if np.abs(total_pred[i] - total_true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(total_true, total_pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 1.57e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 1.55e-3", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 1.82e-3", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 1.57e-3 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

plt.text(11, 15.55, "Symbolic Models", horizontalalignment="center", fontsize=28)

#################################################################################################################
plt.subplot(432).set(xticks=[], yticks=[])

# NN (3vars)

############################### DATA #############################
true = np.log10(np.load("CAMELS_3var_true.npy"))
pred = np.log10(np.load("CAMELS_3var_pred.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '20'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=18)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18)

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 2.08-e3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 7.39e-4", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 9.26e-4", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 8.75e-4 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

plt.text(11, 15.55, "Neural Network (${\sigma}$, $r$, and $v_{max}$)", horizontalalignment="center", fontsize=28)

#################################################################################################################
plt.subplot(433).set(xticks=[], yticks=[])

# NN (11 vars)

############################### DATA #############################
true = np.log10(np.load("true_CAMELS_z0.npy"))
pred = np.log10(np.load("pred_CAMELS_z0.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.0001)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=18)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18)

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 6.87e-4", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 4.17e-4", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 4.27e-4", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 4.42e-4 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

plt.text(11, 15.55, "Neural Network (all 11 properties)", horizontalalignment="center", fontsize=28)

cb_ax = fig.add_axes([.906, .7, .015, .18])
fig.colorbar(plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma"),
             orientation='vertical',cax=cb_ax, ).set_label("Number of Galaxies", rotation=270, fontsize=27, labelpad=25)


######################################################## SIMBA ###################################################################

plt.subplot(434).set(xticks=[])
#  Symbolic Model

############################### DATA #############################
large_true = np.load("new_model_l2_SIMBA_true.npy")
large_pred = np.load("new_model_l2_SIMBA_pred.npy")

med_true = np.load("new_model_m_SIMBA_true.npy")
med_pred = np.load("new_model_m_SIMBA_pred.npy")

small_true = np.load("new_model_s3_SIMBA_true.npy")
small_pred = np.load("new_model_s3_SIMBA_pred.npy")

total_true = np.concatenate((small_true, med_true, large_true))
total_pred = np.concatenate((small_pred, med_pred, large_pred))

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((total_true - total_pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(total_true)):
    num_total += 1
    if np.abs(total_pred[i] - total_true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(total_true, total_pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 3.91e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 1.99e-2", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 4.65e-3", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 1.83e-2 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

#####################################################################################################################
# NN ( 3vars)
plt.subplot(435).set(xticks=[], yticks=[])

############################### DATA #############################
true = np.log10(np.load("SIMBA_3var_true.npy"))
pred = np.log10(np.load("SIMBA_3var_pred.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 3.70e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 4.57e-3", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 2.60e-3", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 4.43e-3 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

#####################################################################################################################
# NN (11vars)
plt.subplot(436).set(xticks=[], yticks=[])

############################### DATA #############################
true = np.log10(np.load("true_SIMBA_z0.npy"))
pred = np.log10(np.load("pred_SIMBA_z0.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 3.74e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 4.82e-3", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 3.67e-3", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 4.70e-3 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

cb_ax2 = fig.add_axes([.906, .508, .015, .18])
fig.colorbar(plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma"),
             orientation='vertical',cax=cb_ax2, ).set_label("Number of Galaxies", rotation=270, fontsize=27, labelpad=25)

######################################################## TNG100 ###################################################################

plt.subplot(437).set(xticks=[])
#  Symbolic Model

############################### DATA #############################
large_true = np.load("new_model_l_TNG100_true.npy")
large_pred = np.load("new_model_l_TNG100_pred.npy")

med_true = np.load("new_model_m_TNG100_true.npy")
med_pred = np.load("new_model_m_TNG100_pred.npy")

small_true = np.load("new_model_s3_TNG100_true.npy")
small_pred = np.load("new_model_s3_TNG100_pred.npy")

total_true = np.concatenate((small_true, med_true, large_true))
total_pred = np.concatenate((small_pred, med_pred, large_pred))

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((total_true - total_pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(total_true)):
    num_total += 1
    if np.abs(total_pred[i] - total_true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(total_true, total_pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 1.36e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 7.28e-4", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 1.44e-3", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 9.48e-4 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

#####################################################################################################################
# NN (3vars)
plt.subplot(438).set(xticks=[], yticks=[])

############################### DATA #############################
true = np.log10(np.load("tng100_3var_true.npy"))
pred = np.log10(np.load("tng100_3var_pred.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 2.35e-2", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 5.41e-4", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 1.07e-3", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 8.12e-3 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

###############################################################################################################
# NN (11vars)
plt.subplot(439).set(xticks=[], yticks=[])

############################### DATA #############################
true = np.log10(np.load("true_tng100_z_0.npy"))
pred = np.log10(np.load("pred_tng100_z_0.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 9.35e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 1.26e-3", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 6.41e-4", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 3.92e-3 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

cb_ax3 = fig.add_axes([.906, .32, .015, .18])
fig.colorbar(plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma"),
             orientation='vertical',cax=cb_ax3, ).set_label("Number of Galaxies", rotation=270, fontsize=27, labelpad=25)

######################################################## TNG300 ###################################################################

plt.subplot(4,3,10).set(xticks=[])
#  Symbolic Model

############################### DATA #############################
large_true = np.load("new_model_l_TNG300_true.npy")
large_pred = np.load("new_model_l_TNG300_pred.npy")

med_true = np.load("new_model_m_TNG300_true.npy")
med_pred = np.load("new_model_m_TNG300_pred.npy")

small_true = np.load("new_model_s3_TNG300_true.npy")
small_pred = np.load("new_model_s3_TNG300_pred.npy")

total_true = np.concatenate((small_true, med_true, large_true))
total_pred = np.concatenate((small_pred, med_pred, large_pred))

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((total_true - total_pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(total_true)):
    num_total += 1
    if np.abs(total_pred[i] - total_true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(total_true, total_pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 1.25e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 1.03e-3", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 1.24e-3", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 1.07e-3 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

#####################################################################################################################
# NN (3vars)
plt.subplot(4,3,11).set(xticks=[], yticks=[])

############################### DATA #############################
true = np.log10(np.load("tng300_3var_true.npy"))
pred = np.log10(np.load("tng300_3var_pred.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 3.92e-3", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 4.94e-4", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 7.44e-4", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 8.80e-4 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

###############################################################################################################
# NN (11vars)
plt.subplot(4,3,12).set(xticks=[], yticks=[])

############################### DATA #############################
true = np.log10(np.load("true_tng300_z_0.npy"))
pred = np.log10(np.load("pred_tng300_z_0.npy"))

mask = [x >= 12 for x in true]
large_true = true[mask]
large_pred = pred[mask]

mask = [ x > 10 and x <12 for x in true]
med_true = true[mask]
med_pred = pred[mask]

mask = [ x <= 10 for x in true]
small_true = true[mask]
small_pred = pred[mask]

############################ ACCURACY ############################
mse_L = np.mean((large_true - large_pred)**2)
mse_M = np.mean((med_true - med_pred)**2)
mse_S = np.mean((small_true - small_pred)**2)
mse_T = np.mean((true - pred)**2)

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(large_true)):
    num_total += 1
    if np.abs(large_pred[i] - large_true[i]) < 0.2:
        num_within += 1
        
percentage_L = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(med_true)):
    num_total += 1
    if np.abs(med_pred[i] - med_true[i]) < 0.2:
        num_within += 1
        
percentage_M = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(small_true)):
    num_total += 1
    if np.abs(small_pred[i] - small_true[i]) < 0.2:
        num_within += 1
        
percentage_S = num_within/num_total * 100

# Calculate The fraction of galaxies whose predicted property is within 0.2 dex of the true 
num_within = 0
num_total = 0

for i in range(np.size(true)):
    num_total += 1
    if np.abs(pred[i] - true[i]) < 0.2:
        num_within += 1
        
percentage_T = num_within/num_total * 100

############################## PLOT ###############################
#plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = '25'

plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma")
#cbar = plt.colorbar(pad=0.007)
#cbar.set_label("Number of Galaxies", rotation=270, fontsize=17, labelpad=25)
#plt.ylabel("Predicted [$log_{10}(M_\odot /h)$]", fontsize=29)
#plt.xlabel("Truth [$log_{10}(M_\odot /h)$]", fontsize=18

# y=x line
min = 6
max = 16
x = np.linspace(min, max, 10)   
plt.plot(x,x, '-r')

plt.ylim(6.4, 15.4)
plt.xlim(6.4, 15.4)

# Vertical lines
plt.plot(np.full((17), 10), np.arange(17), color="black", linestyle = "--")
plt.plot(np.full((17), 12), np.arange(17), color="black", linestyle = "--")

# Text
plt.text(6.6, 15, "MSE: 6.51e-4", fontsize=16)
plt.text(6.55, 14.7, " %.2f"%(percentage_S)+"%", fontsize=16)

plt.text(10.1, 15, "MSE: 3.94e-4", fontsize=16)
plt.text(10.05, 14.7, " %.2f"%(percentage_M)+"%", fontsize=16)

plt.text(12.1, 15, "MSE: 4.35e-4", fontsize=16)
plt.text(12.05, 14.7, " %.2f"%(percentage_L)+"%", fontsize=16)

textstr = "MSE: 4.25e-4 \n%.2f"%(percentage_T)+"%"
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
plt.text(12.25, 7.2, textstr, bbox=props, horizontalalignment="left", fontsize=24)

cb_ax4 = fig.add_axes([.906, .13, .015, .18])
fig.colorbar(plt.hexbin(true, pred, bins="log", mincnt=1, gridsize=220, cmap="plasma"),
             orientation='vertical',cax=cb_ax4, ).set_label("Number of Galaxies", rotation=270, fontsize=27, labelpad=25)

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
