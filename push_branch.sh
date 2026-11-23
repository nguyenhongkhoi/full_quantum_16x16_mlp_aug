#!/usr/bin/env bash
# Dua 1 folder phien ban len 1 branch GitHub (code nam o goc branch), khong dong toi working tree.
# Cach dung: bash push_branch.sh <folder> <branch> "<mo ta ngan>" "<ket qua>" ["commit message"]
# - Branch da co: tao commit moi noi tiep. README.md cua branch = mo ta ngan; README goc -> README_CHI_TIET.md
# - Chi push khi PUSH=1.
set -euo pipefail
cd "$(dirname "$0")"
DIR=$1; BR=$2; DESC=$3; RES=$4; MSG=${5:-"Cap nhat $DIR"}
ROOT=$(pwd)
TMPIDX=$(mktemp); rm -f "$TMPIDX"
export GIT_INDEX_FILE=$TMPIDX
git --work-tree="$ROOT/$DIR" add -A .
# README ngan (KEEP_README=1: giu nguyen README.md cua folder)
if [ "${KEEP_README:-0}" = 1 ]; then :
elif git ls-files --error-unmatch README.md >/dev/null 2>&1; then
  sha=$(git ls-files -s README.md | awk '{print $2}')
  git update-index --force-remove README.md
  git update-index --add --cacheinfo 100644 "$sha" README_CHI_TIET.md
fi
CMD=""; [ -f "$DIR/command.txt" ] && CMD="- Lenh chay day du: \`command.txt\`"
DET=""; git ls-files --error-unmatch README_CHI_TIET.md >/dev/null 2>&1 && DET="- Mo ta chi tiet: \`README_CHI_TIET.md\`"
readme=$(printf '# %s\n\n%s\n\n**Ket qua:** %s\n\n- Folder goc: `%s`\n%s\n%s\n\nMNIST 16x16, seed 0, split-seed 20260911, aug (xoay +-5, dich +-0,05), `--dropout 0`. So tham so bao theo FREE (unitary 2 qubit 16, 1 qubit 4).\n' "$BR" "$DESC" "$RES" "$DIR" "$CMD" "$DET")
if [ "${KEEP_README:-0}" != 1 ]; then
  rsha=$(printf '%s' "$readme" | git hash-object -w --stdin)
  git update-index --add --cacheinfo 100644 "$rsha" README.md
fi
gsha=$(git hash-object -w .gitignore)
git update-index --add --cacheinfo 100644 "$gsha" .gitignore
tree=$(git write-tree)
rm -f "$TMPIDX"; unset GIT_INDEX_FILE
parent=""; git rev-parse -q --verify "refs/heads/$BR" >/dev/null && parent="-p refs/heads/$BR"
if [ -n "$parent" ] && [ "$(git rev-parse "refs/heads/$BR^{tree}")" = "$tree" ]; then echo "$BR: khong doi"; exit 0; fi
c=$(printf '%s\n' "$MSG" | git commit-tree "$tree" $parent)
git update-ref "refs/heads/$BR" "$c"
echo "$BR -> $c ($(git ls-tree -r --name-only "$c" | wc -l) file)"
[ "${PUSH:-0}" = 1 ] && git push -u origin "$BR"
exit 0
