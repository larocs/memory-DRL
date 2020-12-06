import csv

from mysac.envs.pyrep_env import CartPoleEnv
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape

MASS_STABLE_HEIGHT = 0.55
MASS_STABLE_STEPS = 10


class MassState:
    """
    Keeps track of the mass state in the CartPoleEnv.

    The mass is considered stable if its z position is above
    `MASS_STABLE_HEIGHT` for `MASS_STABLE_STEPS` steps.

    Attributes:
        recover_history: keep track of how much steps it takes the mass to
            go back to a stable state after a perturbation

    Args:
        mass: the mass shape in scene
    """

    def __init__(self, mass: Shape, perturbation_mass: Shape):
        self.stable_steps = 0
        self.is_recovering = True
        self.mass = mass
        self.perturbation_masss = perturbation_mass

        self.is_unstable = False

        self.recovering_steps = 0
        self.recover_history = []

        self.unstable_steps = 0
        self.unstable_history = []

    def reset(self):
        """
        Resets the internal state (but keeps the recovery history)
        """
        self.stable_steps = 0
        self.is_recovering = True
        self.recovering_steps = 0

    def step(self) -> bool:
        """
        Computes the current mass state

        Returns:
            True if the mass is stable and should receive a hit from the
            perturbation body.
        """
        _, _, mass_z = self.mass.get_position()
        should_hit = False

        if self.is_recovering:
            if mass_z > MASS_STABLE_HEIGHT:
                self.stable_steps += 1

            # Should transit to the stable state
            if self.stable_steps > MASS_STABLE_STEPS:
                self.is_recovering = False
                should_hit = True
                self.recover_history.append(self.recovering_steps)

                self.stable_steps = 0
                self.recovering_steps = 0

            self.recovering_steps += 1

        else:
            if self.mass.check_collision(self.perturbation_masss) and \
                    not self.is_unstable:
                self.is_unstable = True

            if self.is_unstable:
                self.unstable_steps += 1

            if mass_z < MASS_STABLE_HEIGHT:
                self.is_recovering = True

                self.unstable_history.append(self.unstable_steps)
                self.is_unstable = False
                self.unstable_steps = 0

        return should_hit


class ImpactSliderState:
    """
    Keeps the state of the impact slider.

    Args:
        perturbation_body: a reference to the body that should collide against
            the pole mass
        perturbation_slider: a reference to the joint that controls the
            perturbation_body
        mass: a reference to the pole mass
    """

    def __init__(self, perturbation_body: Shape, perturbation_slider: Joint,
                 mass: Shape):
        self.perturbation_body = perturbation_body
        self.perturbation_slider = perturbation_slider

        self._start_position = self.perturbation_slider.get_position()[1:]

        self.is_going_to_hit = False
        self.velocity = -20

        self.mass = mass

    def reset(self):
        """
        Reset the internal state
        """
        self.is_going_to_hit = False
        self.velocity = -20

    def step(self, should_hit: bool):
        """
        Keeps track of the perturbation_body

        Args:
            should_hit: if true, triggers the joint to hit the pole mass
        """
        mass_x, *_ = self.mass.get_position()

        # Keeps the slider aligned to the mass
        slider_position = [mass_x] + self._start_position
        self.perturbation_slider.set_position(slider_position)
        self.perturbation_slider.set_joint_target_velocity(self.velocity)

        if self.is_going_to_hit:
            if self.perturbation_body.check_collision(self.mass):
                self.is_going_to_hit = False

        elif should_hit:
            self.velocity = - self.velocity
            self.is_going_to_hit = True


class CartPolePerturbationEnv(CartPoleEnv):
    """
    This env has the same MDP of CartPoleEnv, but perturbs the pole when it
    gets into a stable position.
    """
    SCENE_FILE = 'cart_pole_2d_up_perturbation.ttt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.episode = 0

        self.mass_state = MassState(
            mass=self.mass,
            perturbation_mass=Shape(name_or_handle='perturbation_cuboid')
        )

        self.impact_slider_state = ImpactSliderState(
            perturbation_body=Shape(name_or_handle='perturbation_cuboid'),
            perturbation_slider=Joint(name_or_handle='impact_slider'),
            mass=self.mass
        )

        self.history = {
            'mass_z': [],
            'mass_x': [],
            'mass_y': [],
            'hitted': [],
            'should_hit': [],
            'episode': []
        }

    def reset(self, *args, **kwargs):
        """
        Reset the underlying env and 
        """
        self.mass_state.reset()
        self.impact_slider_state.reset()
        self.episode += 1

        return super().reset(*args, **kwargs)

    def append_step(self, should_hit: bool):
        """
        Append the current simulation step to the history
        """
        x, y, z = self.mass.get_position()
        hitted = self.mass.check_collision(
            obj=self.impact_slider_state.perturbation_body)

        self.history['episode'].append(self.episode)
        self.history['mass_x'].append(x)
        self.history['mass_y'].append(y)
        self.history['mass_z'].append(z)
        self.history['hitted'].append(hitted)
        self.history['should_hit'].append(should_hit)

    def step(self, *args, **kwargs):
        """
        Makes a step in the underlying env and updates the env history
        """
        should_hit = self.mass_state.step()
        self.impact_slider_state.step(should_hit=should_hit)

        self.append_step(should_hit=should_hit)

        return super().step(*args, **kwargs)

    def save_eval(self, eval_folder: str):
        """
        Saves the perturbation test history
        """
        with open(eval_folder + '/all_steps.csv', 'w') as csv_file:
            writer = csv.DictWriter(
                f=csv_file,
                fieldnames=['mass_x', 'mass_y', 'mass_z', 'hitted',
                            'should_hit', 'episode']
            )

            for i in range(len(self.history['mass_x'])):
                writer.writerow(
                    {
                        'episode': self.history['episode'][i],
                        'mass_x': self.history['mass_x'][i],
                        'mass_y': self.history['mass_y'][i],
                        'mass_z': self.history['mass_z'][i],
                        'hitted': self.history['hitted'][i],
                        'should_hit': self.history['should_hit'][i],
                    }
                )

        with open(eval_folder + '/recovery_steps.csv', 'w') as csv_file:
            for line in map(str, self.mass_state.recover_history):
                csv_file.write(line + '\n')

        with open(eval_folder + '/unstable_steps.csv', 'w') as csv_file:
            for line in map(str, self.mass_state.recover_history):
                csv_file.write(line + '\n')
