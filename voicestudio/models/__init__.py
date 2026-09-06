from ..backport import install as install_backport


# Runs before the model imports below, several of which import `transformers.models.qwen3_tts` and
# its tokenizers at import time.
install_backport()

from .bigvgan import *
from .breeze_tts import *
from .chroma import *
from .cosyvoice_v1 import *
from .cosyvoice_v2 import *
from .cosyvoice_v3 import *
from .dia import *
from .dia2 import *
from .f5_tts import *
from .higgs_tts2 import *
from .higgs_tts3 import *
from .ommivoice import *
from .parler_tts import *
from .prompt_tts_pp import *
from .qwen3_tts import *
from .spark_tts import *
from .spark_tts_bicodec import *
from .vocos import *
from .vox_instruct import *
