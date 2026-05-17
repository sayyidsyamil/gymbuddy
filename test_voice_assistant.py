import unittest

from voice_assistant import _parse_command, _strip_wake


class VoiceCommandTests(unittest.TestCase):
    def assertVoiceCommand(self, text, expected):
        self.assertEqual(_parse_command(_strip_wake(text)), expected)

    def test_add_reps(self):
        self.assertVoiceCommand("buddy add 10 more reps", ("add_reps", 10))
        self.assertVoiceCommand("buddy 10 more", ("add_reps", 10))
        self.assertVoiceCommand("buddy add ten more rap", ("add_reps", 10))

    def test_reduce_reps(self):
        self.assertVoiceCommand("buddy reduce 5 reps", ("reduce_reps", 5))
        self.assertVoiceCommand("buddy this is too much can u reduce 5 reps", ("reduce_reps", 5))
        self.assertVoiceCommand("buddy minus 12 reps", ("reduce_reps", 12))

    def test_set_specific_target(self):
        self.assertVoiceCommand("buddy set target to 37 reps", ("set_target", 37))
        self.assertVoiceCommand("buddy target 22", ("set_target", 22))
        self.assertVoiceCommand("buddy make it 15 reps", ("set_target", 15))

    def test_session_controls(self):
        self.assertVoiceCommand("buddy can you stop", ("stop", 0))
        self.assertVoiceCommand("buddy start workout", ("start", 0))
        self.assertVoiceCommand("buddy reset", ("reset", 0))


if __name__ == "__main__":
    unittest.main()
