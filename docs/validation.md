# What has been validated, and against what

A simulation that is repeatable is not thereby right. This page lists what the physics and the renderer have been checked against, and
says plainly where there is no reference.

## Physics: checked against first-principles answers (`tests/test_physics_validation.py`)

| behaviour | reference | tolerance |
|---|---|---|
| free fall | z = z0 - g t^2 / 2, v = -g t | 3 cm, 1 % |
| a thrown body | a parabola | 5 cm |
| sliding on a slope | a = g (sin a - mu cos a), for two angles and frictions | 8 % |
| holding on a slope | no sliding below the friction angle | 5 cm in 3 s |
| a bounce | rebound height = e^2 x drop height, for two restitutions | 12 % |
| a collision | momentum conserved; kinetic energy conserved when elastic; equal masses swap velocities | 1 %, 5 % |
| free spin | velocity and spin of a cube unchanged with no force | 0.1 % |
| energy | a dropped pile never gains energy | 5 % noise, no gain |

These found a real defect: Jolt gives every body a default damping of 5 % of its velocity a second, which is drag in a vacuum. It now
defaults to zero.

## Physics: what is NOT validated

- **No comparison with measurements.** The checks above are against analytic answers, not against experiments on real bodies.

## Rendering: checked against the CPU reference

- The CPU and CUDA backends draw the same picture to within 1/255, including normal maps and roughness/metal maps (`tests/test_maps.py`).
- The Vulkan ray-query prototype agrees with the CPU on 99.9996 to 99.9999 % of rays on an NVIDIA GPU, an AMD GPU and a software device.
