# Constants provided
pKa_values = [0.0, 1.5, 2.0, 2.69, 6.13, 10.37]
Ka_values = [10**(-pKa) for pKa in pKa_values]

# Calculate [H+] at pH 3.5
H_concentration = 10**(-3.5)

# Calculate D as given
D = (H_concentration**6 +
     H_concentration**5 * Ka_values[0] +
     H_concentration**4 * Ka_values[0] * Ka_values[1] +
     H_concentration**3 * Ka_values[0] * Ka_values[1] * Ka_values[2] +
     H_concentration**2 * Ka_values[0] * Ka_values[1] * Ka_values[2] * Ka_values[3] +
     H_concentration * Ka_values[0] * Ka_values[1] * Ka_values[2] * Ka_values[3] * Ka_values[4] +
     Ka_values[0] * Ka_values[1] * Ka_values[2] * Ka_values[3] * Ka_values[4] * Ka_values[5])

# Calculate alpha_Y4-
alpha_Y4 = Ka_values[0] * Ka_values[1] * Ka_values[2] * Ka_values[3] * Ka_values[4] * Ka_values[5] / D

print(alpha_Y4)