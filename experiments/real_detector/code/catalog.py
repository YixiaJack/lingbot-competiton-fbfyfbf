"""Object catalog, instruction templates and look-alike groups (no simulator imports, usable in any env)."""
import json
import os
from pathlib import Path

ASSETS = Path(os.environ.get('REAL_ASSETS', Path(__file__).resolve().parent.parent / 'assets'))
META = json.loads((ASSETS / 'assets.json').read_text())
T1_OBJECTS = [k for k, v in META.items() if v['task'] == 1 and not k.startswith('plate_')]
PLATE_COLORS = ['white', 'yellow', 'pink', 'purple', 'blue']
COSMETICS = [k for k, v in META.items() if v['task'] == 2]
T3_OBJECTS = [k for k, v in META.items() if v['task'] == 3]
LEVELS = ['bottom', 'middle', 'top']
CLASSES = T1_OBJECTS + [f'plate_{c}' for c in PLATE_COLORS] + COSMETICS + [f'organizer_{l}' for l in LEVELS] + T3_OBJECTS
NAME = {k: v['name'] for k, v in META.items()}

TEMPLATES = {
    1: ['Put the {o} on the {d} plate.', 'Pick up the {o} and place it on the {d} plate.',
        'Move the {o} onto the {d} plate.', 'Place the {o} on the {d} plate.', 'Could you put the {o} on the {d} plate?'],
    2: ['Put the {o} in the {d} compartment.', 'Place the {o} inside the {d} level of the organizer.',
        'Store the {o} in the {d} compartment.', 'Take the {o} and put it in the {d} section.',
        'Organize the {o} into the {d} compartment.'],
}
TASK3_INSTRUCTION = 'With the brush and dustpan, sweep up all the paper balls and empty the dustpan into the basket.'
# look-alike groups: a third of the scenes draw distractors from the target's group, so recognition is not trivial
CONFUSE = [{'red_apple', 'red_delicious', 'pomegranate', 'tomato', 'persimmon', 'bell_pepper'},
           {'green_apple', 'pear', 'lemon', 'wax_apple', 'starfruit'}, {'orange', 'tangerine', 'persimmon', 'mango', 'pumpkin'},
           {'red_chili', 'green_chili', 'carrot', 'banana'}, {'cucumber', 'green_chili', 'avocado', 'kiwi', 'eggplant'},
           {'strawberry', 'cherry', 'red_apple', 'dragon_fruit'}, {'potato', 'onion', 'kiwi', 'mushroom', 'passion_fruit'},
           {'brush_large', 'brush_small', 'lip_gloss', 'lipstick'}, {'highlighter', 'contour', 'glitter_compact', 'powder_puff'},
           {'lotion_bottle', 'eyeshadow_palette', 'lip_gloss'}]

# demonstrations for the cue experiment: items 0..N1-1 task 1, then N2 task-2 items
# 11/12: the task-2 expert cannot place into the organizer yet (thin brushes slip, tall bottle tips), so the cue A/B
# runs on task 1 only; task 2 stays in the detector and its selection evaluation
N1, N2 = 200, 0
# detector renders: items 0-89 train, 90-109 val, 20 scenes each (1800 + 400 images). Items >= DET_ITEMS are
# never part of the experiment (11/12: the render list still ran to 149 after the cut to 110; those extras are ignored).
DET_TRAIN_ITEMS, DET_ITEMS = 90, 110
