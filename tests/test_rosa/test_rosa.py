#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import unittest

from src.rosa.prompts import RobotSystemPrompts, system_prompts
from src.rosa.rosa import ROSA


class TestROSAPrompts(unittest.TestCase):

    def _bare_rosa(self) -> ROSA:
        # Skip __init__ so no LLM or ROS installation is needed to build prompts
        rosa = ROSA.__new__(ROSA)
        rosa._ROSA__memory_key = "chat_history"
        rosa._ROSA__scratchpad = "agent_scratchpad"
        return rosa

    def test_robot_prompts_do_not_mutate_default_system_prompts(self):
        original = list(system_prompts)

        self._bare_rosa()._get_prompts(
            RobotSystemPrompts(embodiment_and_persona="You are robot A.")
        )

        self.assertEqual(system_prompts, original)

    def test_robot_prompts_do_not_leak_between_instances(self):
        self._bare_rosa()._get_prompts(
            RobotSystemPrompts(embodiment_and_persona="You are robot A.")
        )
        template = self._bare_rosa()._get_prompts(
            RobotSystemPrompts(embodiment_and_persona="You are robot B.")
        )

        rendered = str(template.messages)
        self.assertIn("You are robot B.", rendered)
        self.assertNotIn("You are robot A.", rendered)


if __name__ == "__main__":
    unittest.main()
