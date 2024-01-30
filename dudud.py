# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", device_map="auto")

input = """
Dover (/ˈdoʊvər/ DOH-vər) is a town and major ferry port in Kent, South East England. It faces France across the Strait of Dover, the narrowest part of the English Channel at 33 kilometres (21 mi) from Cap Gris Nez in France. It lies south-east of Canterbury and east of Maidstone. The town is the administrative centre of the Dover District and home of the Port of Dover. Archaeological finds have revealed that the area has always been a focus for peoples entering and leaving Britain. The name derives from the River Dour that flows through it. In recent times the town has undergone transformations with a high-speed rail link to London, new retail in town with St James' area opened in 2018, and a revamped promenade and beachfront. This followed in 2019, with a new 500m Pier to the west of the Harbour, and new Marina unveiled as part of a £330m investment in the area. It has also been a point of destination for many illegal migrant crossings. The Port of Dover provides much of the town's employment, as does tourism including to the landmark White Cliffs of Dover. There were over 368,000 tourists visiting Dover castle in the year of 2019.[3] Dover is classified as a Large-Port Town, due to its large volumes of port traffic and low urban population.[4] History Main article: History of Dover Dover Castle seen from Castle Street. Photograph showing a Dover street scene, c. 1860 Archaeological finds have shown that there were Stone Age people in the area, and that some Iron Age finds also exist.[5] During the Roman period, the area became part of the Roman communications network. It was connected by road to Canterbury and Watling Street and it became Portus Dubris, a fortified port. Dover has a partly preserved Roman lighthouse (the tallest surviving Roman structure in Britain) and the remains of a villa with preserved Roman wall paintings.[6] Dover later figured in Domesday Book (1086). Forts were built above the port and lighthouses were constructed to guide passing ships. It is one of the Cinque Ports.[7] and has served as a bastion against various attackers: notably the French during the Napoleonic Wars and Germany during the Second World War. During the Cold War, a Regional Seat of Government was located within the White Cliffs beneath Dover Castle. This is omitted from the strategic objects appearing on the Soviet 1:10,000 city plan of Dover that was produced in 1974.[8] The port would have served as an embarkation point for sending reinforcements to the British Army of the Rhine in the event of a Soviet ground invasion of Europe. In 1974 a discovery was made at Langdon Bay off the coast near Dover. It contained bronze axes of French design and is probably the remainder of the cargo of a sunken ship. At the same time, this find also shows that trade routes across the Channel between England and France existed already in the Bronze Age, or even earlier. In 1992, the so-called Dover boat from the Bronze Age was discovered in six metres depth underwater. This is one of the oldest finds of a seaworthy boat. Using the radiocarbon method of investigation, the boat's construction was dated to approximately 1550 BC. Etymology First recorded in its Latinised form of Portus Dubris, the name derives from the Brythonic word for water (dwfr in Middle Welsh, dŵr in Modern Welsh apart from 'dwfrliw' (Watercolour) which has retained the old Welsh spelling, dour in Breton). The same element is present in the town's French name Douvres and the name of the river, Dour, which is also evident in other English towns such as Wendover. However, the modern Modern Welsh name Dofr is an adaptation of the English name Dover.[9] The current name was in use at least by the time of Shakespeare's King Lear (between 1603 and 1606), in which the town and its cliffs play a prominent role.[10] The Siege of Dover (1216) Main article: Battle of Sandwich (1217) Louis VIII of France landed his army, seeking to depose King Henry III, on Dover's mainland beach. Henry III ambushed Louis' army with approximately 400 bowmen atop The White Cliffs of Dover and his cavalry attacking the invaders on the beach. However, the French slaughtered the English cavalry and made their way up the cliffs to disperse the bowmen. Louis' army seized Dover village, forcing the English back to Canterbury. French control of Dover lasted for three months after which English troops pushed back, forcing the French to surrender and return home.[citation needed] Geography and climate 1945 Ordnance Survey map of Dover, showing the harbour Dover is in the south-east corner of Britain. From South Foreland, the nearest point to the European mainland, Cap Gris Nez is 34 kilometres (21 mi) away across the Strait of Dover.[11] The site of its original settlement lies in the valley of the River Dour, sheltering from the prevailing south-westerly winds. This has led to the silting up of the river mouth by the action of longshore drift. The town has been forced into making artificial breakwaters to keep the port in being. These breakwaters have been extended and adapted so that the port lies almost entirely on reclaimed land. The higher land on either side of the valley – the Western Heights and the eastern high point on which Dover Castle stands – has been adapted to perform the function of protection against invaders. The town has gradually extended up the river valley, encompassing several villages in doing so. Little growth is possible along the coast, since the cliffs are on the sea's edge. The railway, being tunnelled and embanked, skirts the foot of the cliffs. Dover has an oceanic climate (Köppen classification Cfb) similar to the rest of the United Kingdom with mild temperatures year-round and a light amount of rainfall each month. The warmest recorded temperature was 37.4 °C (99.3 °F), recorded at Langdon Bay on 25 July 2019,[12] Whilst the lowest recorded temperature was −9.5 °C (14.9 °F), recorded at Dover RMS on 31 January 1972.[13] The temperature is usually between 3 °C (37 °F) and 21.1 °C (70.0 °F).
"""

while True:
    input_ids = tokenizer.encode(input, return_tensors="pt").cuda()
    output = model.generate(input_ids, max_length=4000, do_sample=True, top_p=0.95, top_k=60)
    print(tokenizer.decode(output[0]))