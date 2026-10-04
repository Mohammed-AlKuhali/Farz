#!/bin/bash
# Text-to-speech PLACEHOLDER clips for local testing, made with macOS `say` (voice "Majed", ar_001) + ffmpeg.
# They go to app/audio-placeholder/ and are only built in with VITE_AUDIO=placeholder (see app/public/audio/README.md).
# A real voice pack (recordings under the same 19 filenames, docs/CLIPS.md) goes in app/public/audio/ instead.
set -euo pipefail
mkdir -p "$(dirname "$0")/../audio-placeholder"
cd "$(dirname "$0")/../audio-placeholder"
TMP=$(mktemp -d)
while IFS='|' read -r id text; do
  [ -z "$id" ] && continue
  if [ -n "${ONLY:-}" ] && [ "$id" != "$ONLY" ]; then continue; fi
  say -v Majed -r 165 -o "$TMP/$id.aiff" "$text"
  ffmpeg -loglevel error -y -i "$TMP/$id.aiff" -ac 1 -ar 22050 -codec:a libmp3lame -b:a 48k "$id.mp3"
  echo "$id.mp3"
done <<'LIST'
c01_welcome|أهلين. صوّري حوالي مية حبة بن أخضر على قماشة بيضا، مفرّقة عن بعض.
c02_blur|الصورة مش واضحة. ثبّتي التلفون وصوّري ثاني.
c03_dark|الصورة مظلمة. صوّري في مكان فيه نور.
c04_spread|الحبوب لاصقة ببعض. فرّقيها وصوّري ثاني.
c05_count|حطّي حوالي مية حبة، وصوّري ثاني.
c06_not_green|هذا مش بن أخضر. صوّري البن بعد التقشير وقبل التحميص.
c07_unsure|مش متأكد. ودّي العينة للجمعية يشوفها المختص.
c08_clean|البن نظيف، العيوب قليلة.
c09_some|فيه عيوب. شيلي الحبوب المعلّمة بالأحمر وصوّري ثاني.
c10_many|العيوب كثيرة. فرّزي البن كله قبل البيع.
c11_dark_beans|فيه حبوب سودا أو حامضة. شيليها قبل البيع.
c12_insect|فيه حبوب مخرّمة. ورّيها الجمعية.
c13_broken|فيه حبوب مكسّرة. شيليها قبل البيع.
c14_unhulled|فيه حب لسه بقشره. قشّري العينة قبل الفحص.
c15_slip|الرسالة فيها عدد الحبوب والعيوب وتاريخ اليوم، ومكتوب فيها: فحص ذاتي مش تصنيف. ترسليها؟
c15b_slip_unsure|الرسالة فيها عدد الحبوب وتاريخ اليوم، ومكتوب فيها: مش متأكد، فحص ذاتي مش تصنيف. ترسليها؟
c16_privacy|الصورة انمسحت. ما يطلع من التلفون إلا الرسالة اللي ترسليها أنتي.
c17_wiped|انمسح كل شي.
c18_another|النتيجة قريبة. صوّري حفنة ثانية من نفس البن عشان نتأكد.
LIST
rm -rf "$TMP"
