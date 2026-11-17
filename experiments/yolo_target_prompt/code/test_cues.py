import unittest
from cues import apply_cue, parse_instruction


class Frame:
    shape=(240,320,3)


class CueContract(unittest.TestCase):
    def test_parser_rejects_unknown_instructions(self):
        self.assertIsNone(parse_instruction('Pick a tomato.'))
        self.assertEqual(parse_instruction('Pick up the blue block and place it on the yellow plate.'),(2,3))

    def test_coordinates_are_xyxy_normalized_to_original_camera(self):
        image=Frame()
        _,text=apply_cue(image,'Pick.',[32,24,64,48],[160,120,320,240],'coords')
        self.assertEqual(text,'Pick. Target box [0-1000]: (100,100,200,200). Destination box: (500,500,1000,1000).')

    def test_missing_either_detection_preserves_original_input(self):
        image=Frame()
        for mode in ('text','box','coords'):
            for source,dest in ((None,[1,2,3,4]),([1,2,3,4],None),(None,None)):
                output,text=apply_cue(image,'Original.',source,dest,mode)
                self.assertIs(output,image)
                self.assertEqual(text,'Original.')


if __name__=='__main__': unittest.main()
