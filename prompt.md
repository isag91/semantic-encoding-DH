# Digital Literature Dataset - LLM Description Generation Prompt (inspired by the ARISE pipeline)

You are an expert in digital literature classification. Generate a comprehensive description for a categorical feature value. Do try to be as semantically dscriminative as possible.

For EACH feature-value pair below, generate a description following this EXACT format:

[CORE] 1-2 sentences: Provide a general definition of the value (not necessarily in the context of digital art).
[INDICATOR] 1-2 sentences: What characteristics this value indicates in the context of a work of digital literature/art.

IMPORTANT:
- Use CONCRETE, STRAIGHTFORWARD descriptions
- NO abbreviations or acronyms
- NO obscure jargon - descriptions must be clear and understandable
- Give specific examples where helpful

Here are ALL the feature-value pairs that need descriptions:


FEATURE: access_hardware 
- Value 0: computer
- Value 1: printer
- Value 2: internet browser
- Value 3: loudspeakers
- Value 4: headphones
- Value 5: smartphone
- Value 6: camera
- Value 7: tablet
- Value 8: CD player
- Value 9: motion capture

FEATURE: publication_type
- Value 0: application
- Value 1: file
- Value 2: website
- Value 3: platform
- Value 4: CD/DVD
- Value 5: exhibition
- Value 6: print
- Value 7: installation
- Value 8: software
- Value 9: social media
- Value 10: virtual world

FEATURE: genre
- Value 0: poetry
- Value 1: narrative
- Value 2: poetry and narrative
- Value 3: other

FEATURE: technical_requirements
- Value 0: internet connection
- Value 1: install software/add on
- Value 2: internet browser
- Value 3: RAM
- Value 4: execute locally
- Value 5: submit to emulation
- Value 6: access to platform / service

FEATURE: format
- Value 0: image
- Value 1: video game
- Value 2: text
- Value 3: virtual environment
- Value 4: video
- Value 5: audio
- Value 6: database
- Value 7: search engine
- Value 8: physical artefact
- Value 9: access to platform/service

FEATURE: program
- Value 0: Adobe Acrobat
- Value 1: Bitsy
- Value 2: bot
- Value 3: QR code
- Value 4: EXE
- Value 5: Flash
- Value 6: GIF
- Value 7:  HTML/PHP
- Value 8: Artificial Intelligence
- Value 9: Java
- Value 10: Macromedia
- Value 11: Microsoft Office
- Value 12: Midipoet
- Value 13: Does not apply
- Value 14: eko platform
- Value 15: No information
- Value 16: Video
- Value 17: Virtual Reality Modelling Language
- Value 18: ASCII
- Value 19: Minitel
- Value 20: Fortran
- Value 21: Unity
- Value 22: jGnoetry
- Value 23: Asymetrix Toolbox
- Value 24: Jasc Animation Shop
- Value 25: Managana
- Value 26: Paint Shop Pro


FEATURE: reading_process (these refer to the ways in which art artowrk is interacted with, i.e. the reader as the manipulate elements on the screen, observe them, navigate betweek them, activate some features etc/)
- Value 0: navigation
- Value 1: manipulation
- Value 2: observation
- Value 3: alter elements
- Value 4: content generation
- Value 5: activate / deactivate
- Value 6: detect input devices
- Value 7: selection of elements
- Value 8: log in
- Value 9: upload files
- Value 10: download content
- Value 11: input information

FEATURE: program
- Value 0: Adobe Acrobat
- Value 1: Bitsy
- Value 2: bot
- Value 3: QR code
- Value 4: EXE
- Value 5: Flash
- Value 6: GIF
- Value 7:  HTML/PHP
- Value 8: Artificial Intelligence
- Value 9: Java
- Value 10: Macromedia
- Value 11: Microsoft Office
- Value 12: Midipoet
- Value 13: Does not apply
- Value 14: eko platform
- Value 15: No information
- Value 16: Video
- Value 17: Virtual Reality Modelling Language
- Value 18: ASCII
- Value 19: Minitel
- Value 20: Fortran
- Value 21: Unity
- Value 22: jGnoetry
- Value 23: Asymetrix Toolbox
- Value 24: Jasc Animation Shop
- Value 25: Managana
- Value 26: Paint Shop Pro



OUTPUT FORMAT: Return a JSON object where each key is "feature_value" (e.g., "access_hardware_0", "access_hardware_1") and the value is the complete description string.

{
  "access_hardware_0": "[CORE] ... [INDICATOR] ... ",
  "access_hardware_1": "[CORE] ... [INDICATOR] ... [ ",
  ...
}

Generate descriptions for all feature-value pairs listed above.
