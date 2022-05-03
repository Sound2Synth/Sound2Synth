from .audio_utils import *
try:
    sys.path.append("./PyAU/")
    from pyau.engines import AudioEngine, MusicScheduler
except Exception as e:
    print("Warning: PyAU not supported on this environment!")
    print(e)

class Synthesizer(object):
    def __init__(self,
        name="AUSampler",
        version=f"_{FORMATTED_TIME()}",
        render_mode=True,
        effects_name = list(),
        use_mixer=True,
        midi_settings_dir=get_config('midi_settings_dir'),
        temp_output_dir=get_config('temp_output_dir'),
        audio_output_dir=get_config('audio_output_dir'),
        clear_cache_on_exit=True,
        debug=False,
    ):
        self.name = name
        self.version = version
        self.render_mode = render_mode
        self.effects_name = effects_name
        self.use_mixer = use_mixer
        self.midi_settings_dir = midi_settings_dir
        self.temp_output_dir = pjoin(temp_output_dir,self.name+self.version)
        CreateFolder(self.temp_output_dir)
        self.audio_output_dir = pjoin(audio_output_dir,self.name+self.version)
        CreateFolder(self.audio_output_dir)
        self.clear_cache_on_exit = False if debug else clear_cache_on_exit
        self.debug = debug

    def __enter__(self):
        # Starting AudioEngine
        self.engine = AudioEngine(render_mode=self.render_mode)
        self.engine.__enter__()
        # Try to set the synthesizer, effects, mixer, and music scheduler
        try:
            # Check the existence of synthesizer and effects
            assert (self.name in self.engine.scanner.available_instruments), \
            f"The provided synthesizer '{self.name}' does not exist! Available instruments: {self.engine.scanner.available_instruments}."

            for effect_name in self.effects_name:
                assert (effect_name in self.engine.scanner.available_effects), \
                f"The provided effect '{effect_name}' does not exist! Available effects: {self.engine.scanner.available_effects}."

            # Register synthesizer
            self.synth = self.engine.pool.register_au_instrument(self.name); self.output = self.synth
            
            # Register Effects
            self.effects = []
            for effect_name in self.effects_name:
                self.effects.append(self.engine.pool.register_effect(effect_name))
                self.output.connect(self.effects[-1], from_bus=0, to_bus=0)
                self.output = self.effects[-1]
            
            # Link to mixer channel or directly link to the output channel
            if self.use_mixer:
                self.output.connect(self.engine.pool.main_mixer_node, from_bus=0, to_bus=0)
            else:
                self.output.connect(self.engine.pool.output_node, from_bus=0, to_bus=0)
            
            # Visualize for debugging
            if self.debug:
                self.engine.pool.show()

            # Start Engine
            self.engine.start_hardware()

            # Start Renderer
            self.renderer = MusicScheduler(self.engine)
            self.renderer.__enter__()

            self.parameters = self.synth.global_parameters()
            return self
        except AssertionError as e:
            self.engine.__exit__(*sys.exc_info())
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clear_cache_on_exit:
            self.clear_cache(delete=True)
        self.renderer.__exit__(exc_type, exc_val, exc_tb)
        self.engine.__exit__(exc_type, exc_val, exc_tb)
    
    def render(self, midi_setting_file = f"{get_config('default_midi_settings_args')['default_pitch']}.mid", no_cache=True):
        audio_output_file = RandString(16)+".wav"; temp = True,
        self.renderer.midi = mido.MidiFile(pjoin(self.midi_settings_dir,midi_setting_file))
        default_midi_settings_args = get_config('default_midi_settings_args')
        self.renderer.render(pjoin(self.temp_output_dir,audio_output_file), start=0,
                             end=default_midi_settings_args['recording_beats']/default_midi_settings_args['bpm']*60)
        audio, sample_rate = tau.load(pjoin(self.temp_output_dir,audio_output_file))
        target_sample_rate = get_config('default_torchaudio_args')['sample_rate']
        if sample_rate!=target_sample_rate:
            audio = tau.transforms.Resample(orig_freq=sample_rate,new_freq=target_sample_rate)(waveform=audio)
            sample_rate = target_sample_rate
        if no_cache:
            Delete(pjoin(self.temp_output_dir,audio_output_file), rm=True)
        return audio, sample_rate

    def clear_cache(self, delete=False):
        Delete(self.temp_output_dir, rm=True) if delete else ClearFolder(self.temp_output_dir, rm=True)
    
    def load_plist(self, path):
        with open(path, 'r') as f:
            self.synth.states = f.read()

class BaseInterface(object):
    ordered_descriptors = MemberDict()

    def __init__(self, data):
        self.data = MemberDict(data)
    
    @classmethod
    def from_synth(cls, synth):
        normalize_function = {
            key:(lambda x:(x-param.min_value)/(param.max_value-param.min_value)) for key,param in synth.parameters.items()
        }
        return cls({
            key:normalize_function[key](param.current_value) for key, param in synth.parameters.items()
        })

    @classmethod
    def from_dict(cls, data):
        return cls(data)
    
    @classmethod
    def from_tensor(cls, data):
        instance = cls()
        for i,key in enumerate(cls.ordered_descriptors):
            instance.data[key] = float(data[i])
        return instance
    
    def to_synth(self, synth):
        restore_function = {
            key:(lambda x:x*(param.max_value-param.min_value)+param.min_value) for key,param in synth.parameters.items()
        }
        for key in restore_function:
            synth.parameters[key].current_value = restore_function[key](self.data[key])

    def to_dict(self):
        return MemberDict(self.data)

    def to_tensor(self):
        values = [self.data[key] for key in self.__class__.ordered_descriptors]
        return torch.tensor(values, dtype=torch.float)
