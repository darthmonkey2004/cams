#!/bin/bash


html_dir="$HOME/.local/lib/python3.6/site-packages/NicVision/server/templates"
index_html="$html_dir/index.html"
if [ -f "$index_html" ]; then
	mv "$index_html" "$index_html.bak"
	touch "$index_html"
fi
start_html="<!doctype html>
<html lang=\"en\">
<head>
	<!-- Required meta tags -->
	<meta charset=\"utf-8\">
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=yes\">

	<!-- Bootstrap CSS -->
	<link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css\"
		  integrity=\"sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO\" crossorigin=\"anonymous\">

	<title>NicVision Camera Stream Viewer</title>
"
echo "$start_html" > "$index_html"
readarray cameras <<< $(python3 -c "import cams; cameras = cams.readConfToShell(); print (cameras)" | grep -v "None")
ct="${#cameras[@]}"
idx=0
functions=()
if [ -f "temp.txt" ]; then
	rm temp.txt
fi
idxs=()
for cam in ${cameras[@]}; do
	idxs+=($idx)
	idx=$(( idx + 1 ))
	f="openWin$idx()"
	path="/cam/$idx/"
	code="
		function f() {
			window.open('$path');
		}"
	echo "$code" >> temp.txt
done
functions=$(cat temp.txt)
rm temp.txt
script_block="
	<script>
$functions
	</script>"

echo "$script_block" >> "$index_html"
start_table="</head>
<body>
	<table border="2" height="100%" width="100%">
		<tbody>
			<tr>
"
echo "$start_table" >> "$index_html"
pos=0
for id in ${idxs[@]}; do
	pos=$(( pos + 1 ))
	if [ "$pos" != "3" ]; then
		img_html="				<td><div><button onclick=\"openWin$id()\"><img src=\"{{ url_for(\"video_feed\", id=\"$id\") }}\" width=\"100%\" height=\"100%\"></button></div></td>
"
	elif [ "$pos" = "3" ]; then
		pos=0
		img_html="			</tr>
			<tr>
				<td><div><button onclick=\"openWin$id()\"><img src=\"{{ url_for(\"video_feed\", id=\"$id\") }}\" width=\"100%\" height=\"100%\"></button></div></td>
"
	fi
	echo "$img_html" >> "$index_html"
done
html_end="			</tr>
		</tbody>
	</table>
</body>
</html>"
echo "$html_end" >> "$index_html"
for id in ${idxs[@]}; do
	html="<!doctype html>
<html lang=\"en\">
<head>
	<!-- Required meta tags -->
	<meta charset="utf-8">
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=yes\">

	<!-- Bootstrap CSS -->
	<link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css\"
		  integrity=\"sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO\" crossorigin=\"anonymous\">

	<title>Camera $id</title>
</head>
<body>
	<div><img src=\"{{ url_for(\"video_feed\", id=\"$id\") }}\" width=\"100%\" height=\"100%\"></div>
</body>
</html>"
	echo "$html" > "$html_dir/$id.html"
done




echo "Web pages created for ${#idxs[@]} cameras."
exit 0
