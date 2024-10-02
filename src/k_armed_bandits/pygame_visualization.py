import sys
from typing import Tuple

import pygame

from src.k_armed_bandits.bandits import FixedValueBandits, FixedBanditsConfig, BanditsType, GaussianValueBandits, \
    GaussianBanditsConfig
from src.utils.randomizer import DefaultUniformRandomizer, DefaultNormalRandomizer

# Layout constants
WINDOW_HEIGHT = 200
BANDITS_WIDTH = 80
BANDITS_HEIGHT = 120
BACKGROUND_COLOR = (255, 255, 255)
BANDIT_COLOR = (0, 0, 0)
BUTTON_COLOR = (255, 0, 0)
BUTTON_RADIUS = 30
BUTTON_TEXT_COLOR = (0, 0, 0)
REWARD_TEXT_COLOR = (255, 255, 255)
UI_TEXT_COLOR = (0, 0, 0)


pygame.font.init()
font = pygame.font.SysFont('Comic Sans MS', 25)


class Button(pygame.sprite.Sprite):
    """Sprite to represent a single bandit action"""

    def __init__(self, groups: pygame.sprite.AbstractGroup, bandit_num: int):
        super().__init__(groups)
        self.bandit_num = bandit_num
        self.image = pygame.Surface((BUTTON_RADIUS * 2, BUTTON_RADIUS * 2), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(bandit_num * BANDITS_WIDTH + BANDITS_WIDTH // 2,
                                                BANDITS_WIDTH // 2))

    def handle_click(self) -> int:
        """Returns the bandit id"""
        return self.bandit_num

    def update(self):
        """Update sprite"""
        pygame.draw.circle(self.image, BUTTON_COLOR, (self.rect.width // 2, self.rect.width // 2), BUTTON_RADIUS)

        text_surface = font.render(f"{self.bandit_num}", True, BUTTON_TEXT_COLOR)
        text_surface_rect = text_surface.get_rect()

        text_surface_x = (self.image.get_width() - text_surface_rect[2]) // 2
        text_surface_y = (self.image.get_height() - text_surface_rect[3]) // 2

        self.image.blit(text_surface, dest=(text_surface_x, text_surface_y))


class RewardText(pygame.sprite.Sprite):
    """Sprite to show the reward from a bandit"""

    def __init__(self, groups: pygame.sprite.AbstractGroup, bandit_num: int):
        super().__init__(groups)
        self.bandit_num = bandit_num
        self.reward = ""

        self.image = pygame.Surface((BANDITS_WIDTH - 2, BANDITS_HEIGHT - BANDITS_WIDTH - 1), pygame.SRCALPHA)
        self.rect = self.image.get_rect(topleft=(bandit_num * BANDITS_WIDTH + 1, BANDITS_WIDTH))

    def update_reward(self, value: float) -> None:
        """Change stored reward value"""
        self.reward = str(round(value, 2))

    def clear_reward(self) -> None:
        """Clean stored reward value"""
        self.reward = ""

    def update(self):
        """Update sprite"""
        self.image.fill(BANDIT_COLOR)
        text_surface = font.render(self.reward, True, REWARD_TEXT_COLOR)
        text_surface_rect = text_surface.get_rect()
        text_surface_x = (self.image.get_width() - text_surface_rect[2]) // 2
        text_surface_y = (self.image.get_height() - text_surface_rect[3]) // 2

        self.image.blit(text_surface, dest=(text_surface_x, text_surface_y))


class UI(pygame.sprite.Sprite):
    """Sprite that shows the accumulated score and the step number"""

    def __init__(self, groups: pygame.sprite.AbstractGroup, num_bandits: int):
        super().__init__(groups)
        self.score = 0
        self.step = 0

        self.image = pygame.Surface((BANDITS_WIDTH * num_bandits, WINDOW_HEIGHT - BANDITS_HEIGHT), pygame.SRCALPHA)
        self.rect = self.image.get_rect(topleft=(0, BANDITS_HEIGHT))
    
    def update_score(self, reward: float) -> None:
        self.score = round(reward + self.score, 2)
    
    def update_step(self, step_value: int) -> None:
        self.step = step_value
    
    def update(self):
        """Update sprite"""

        self.image.fill(BACKGROUND_COLOR)

        score_text_surface = font.render(f"Score: {self.score}", True, UI_TEXT_COLOR)
        score_text_surface_rect = score_text_surface.get_rect()
        score_text_surface_x = 30
        score_text_surface_y = (self.image.get_height() - score_text_surface_rect[3]) // 2

        self.image.blit(score_text_surface, dest=(score_text_surface_x, score_text_surface_y))

        step_text_surface = font.render(f"Step: {self.step}", True, UI_TEXT_COLOR)
        step_text_surface_rect = step_text_surface.get_rect()
        step_text_surface_x = 240
        step_text_surface_y = (self.image.get_height() - step_text_surface_rect[3]) // 2

        self.image.blit(step_text_surface, dest=(step_text_surface_x, step_text_surface_y))


class GameEngine:

    def __init__(self, bandits_env: BanditsType):
        pygame.init()
        self.bandits_env = bandits_env
        window_size = (BANDITS_WIDTH * self.bandits_env.num_bandits, WINDOW_HEIGHT)
        self.display_surface = pygame.display.set_mode(window_size)
        pygame.display.set_caption(f"{self.bandits_env.__class__.__name__}")
        self.bg = pygame.Surface(window_size)
        self._set_background()

        # Sprites groups setup
        self.all_sprites = pygame.sprite.Group()

        # Generate game
        self.buttons = [Button(self.all_sprites, i) for i in range(self.bandits_env.num_bandits)]
        self.texts = [RewardText(self.all_sprites, i) for i in range(self.bandits_env.num_bandits)]
        self.ui = UI(self.all_sprites, self.bandits_env.num_bandits)

    def _set_background(self):
        """Draw background"""
        self.bg.fill(BACKGROUND_COLOR)
        for i in range(self.bandits_env.num_bandits):
            pygame.draw.rect(self.bg, BANDIT_COLOR, (i * BANDITS_WIDTH + 1, 1, BANDITS_WIDTH - 2, BANDITS_HEIGHT - 2))
    
    def _handle_click(self, click_pos: Tuple[int, int]):
        """Update game state after a click on a bandit"""
        clicked_bandit = [b.bandit_num for b in self.buttons if b.rect.collidepoint(click_pos)]
        if len(clicked_bandit) > 0:
            clicked_bandit = clicked_bandit[0]
            _, reward, _ = self.bandits_env.step(clicked_bandit)

            for i, reward_text in enumerate(self.texts):
                if i == clicked_bandit:
                    reward_text.update_reward(reward)
                else:
                    reward_text.clear_reward()

            self.texts[clicked_bandit].update_reward(reward)
            
            self.ui.update_score(reward)
            self.ui.update_step(self.bandits_env.step_num)

            print(clicked_bandit)

    def run(self):
        """Runs a bandits game"""

        while True:

            # Events loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            # Update game
            self.all_sprites.update()

            # Draw frame
            self.display_surface.blit(self.bg, (0, 0))
            self.all_sprites.draw(self.display_surface)

            # Update window
            pygame.display.update()


def main():

    # bandits_type = "fixed_values"
    bandits_type = "gaussian_values"

    if bandits_type == "fixed_values":
        f_bandits = FixedValueBandits(FixedBanditsConfig(max_steps=20, seed=0),
                                      DefaultUniformRandomizer())
        ge = GameEngine(f_bandits)
        ge.run()
    elif bandits_type == "gaussian_values":
        g_bandits = GaussianValueBandits(GaussianBanditsConfig(max_steps=20, seed=0),
                                         DefaultNormalRandomizer())
        ge = GameEngine(g_bandits)
        ge.run()
    else:
        raise ValueError("Bandits environment type not found")


if __name__ == '__main__':
    main()
