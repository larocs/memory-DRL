STATES_TO_DROP = dict(
    head_z=slice(0, 1),
    orientation_x=slice(1, 2),
    orientation_y=slice(2, 3),
    orientation_z=slice(3, 4),
    linear=slice(4, 7),
    angular=slice(7, 10),
    joint_positions=slice(10, 21)
)
