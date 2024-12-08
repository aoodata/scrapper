function formatScore(score) {
    // add commas to the score
    return score.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

async function RankingDataProxy() {
    let RankingData = {};

    RankingData["data"] = await pywebview.api.ranking("get_ranking");
    RankingData["commandersDict"] = await pywebview.api.ranking("get_commanders_dict");

    let data = RankingData["data"];

    RankingData["merge"] = async function (commanders_list) {
        let new_name = await pywebview.api.ranking("merge", commanders_list);
        for (let i = 1; i < commanders_list.length; i++) {
            Object.assign(data[new_name], data[commanders_list[i]]);
            delete data[commanders_list[i]];
        }
        data[new_name]["diagnostic"]=  {
            "warning_level": 0,
            "most_similars": [],
            "warnings": []
        }
        return new_name;
    }

    RankingData["split"] = async function (commander, rankings) {
        let new_name = await pywebview.api.ranking("split", commander, rankings);
        data[new_name] = {};
        for (let i = 0; i < rankings.length; i++) {
            data[new_name][rankings[i]] = data[commander][rankings[i]];
            delete data[commander][rankings[i]];
        }
        return new_name;
    }

    /*    def delete(self, commander):
        del self.ranking[commander]*/
    RankingData["delete"] = async function (commander) {
        await pywebview.api.ranking_data("delete", commander);
        delete data[commander];
    }

    RankingData["commanders_sorted_by_warning"] = function () {
        let commanders = Object.keys(data);
        commanders.sort(function (a, b) {
            return data[b]["diagnostic"]["warning_level"] - data[a]["diagnostic"]["warning_level"];
        });
        return commanders;
    }

    return RankingData;
}

async function FusionDataProxy() {
    let FusionData = {};

    FusionData["data"] = await pywebview.api.fusion("get_data");
    let data = FusionData["data"];
    data["commanders_to_insert"] = new Set(data["commanders_to_insert"]);
    let matched_commanders_to_insert = data["matched_commanders_to_insert"];
    let commanders_to_insert = data["commanders_to_insert"];
    let matching_stats = data["matching_stats"];
    let commander_to_insert_scores = data["commander_to_insert_scores"];
    let commander_db_scores = data["commander_db_scores"];
    let non_matched_commanders_db = data["non_matched_commanders_db"];

    FusionData["add_matched_commander"] = async function (commander_name_in_ranking_data, commander_name_in_db, score_diff) {
        await pywebview.api.fusion("add_matched_commander", commander_name_in_ranking_data, commander_name_in_db, score_diff);
        matched_commanders_to_insert[commander_name_in_ranking_data] = [commander_name_in_db, score_diff];
        commanders_to_insert.delete(commander_name_in_ranking_data);
    }

    FusionData["remove_matched_commander"] = async function (commander_name_in_ranking_data) {
        await pywebview.api.fusion("remove_matched_commander", commander_name_in_ranking_data);
        delete matched_commanders_to_insert[commander_name_in_ranking_data];
        commanders_to_insert.add(commander_name_in_ranking_data);
    }
    return FusionData;
}


//commander_reputation:16750 vs 15650 (6.57)
//commander_kill:48557628 vs 48557628 (0.00)
//commander_loss:2594590 vs 2594590 (0.00)
//commander_island:2841 vs 2841 (0.00)
//commander_merit:227493 vs 223914 (1.57)
//commander_city:37 vs 37 (0.00)
//commander_officer:8133303 vs 8080626 (0.65)
//commander_titan:1623196 vs 1622396 (0.05)
//commander_warplane:436619 vs undefined (NaN)
//commander_level:62 vs 62 (0.00)
//commander_warplane:436619 vs ---

let displayed_rankings = new Set();
displayed_rankings.add("commander_reputation");
displayed_rankings.add("commander_kill");
displayed_rankings.add("commander_loss");
displayed_rankings.add("commander_island");
displayed_rankings.add("commander_merit");
displayed_rankings.add("commander_city");
displayed_rankings.add("commander_officer");
displayed_rankings.add("commander_titan");
displayed_rankings.add("commander_warplane");
displayed_rankings.add("commander_level");

let ranking_names = {
    "commander_reputation": "Reputation",
    "commander_kill": "Kill",
    "commander_loss": "Loss",
    "commander_island": "Island",
    "commander_merit": "Merit",
    "commander_city": "City",
    "commander_officer": "Officer",
    "commander_titan": "Titan",
    "commander_level": "Level",
    "commander_warplane": "Warplane",
    "commander_power": "Power",
}

function create_commander_description_element(commander_name, commander_scores) {
    let commander_description = document.createElement("div");
    let commander_alliance = commander_scores["alliance"];
    commander_scores = commander_scores["scores"];
    commander_description.style.width = "50%";
    commander_description.style.display = "inline-block";
    let commander_name_element = document.createElement("div");
    commander_name_element.innerHTML = "<b>" + commander_name + "(" + commander_alliance + ")</b>";
    commander_name_element.style.backgroundColor = "lightblue";
    commander_description.appendChild(commander_name_element);
    for (let score_name in commander_scores) {
        let score = commander_scores[score_name];
        commander_description.innerHTML += "<div style='padding-left: 12px;'><b>" +  ranking_names[score_name] + '</b>:' + score + "</div>";
    }
    return commander_description;

}

function create_commander_score_comparison_element(commander_name_in_ranking, commander_scores_in_ranking, commander_name_in_db, commander_scores_in_db, score_diff) {
    let commander_score_comparison_element = document.createElement("div");
    commander_score_comparison_element.className = "commander_score_comparison_element";
    commander_score_comparison_element.style.display = "inline-block";

    let commander_alliance_in_ranking = commander_scores_in_ranking["alliance"];
    let commander_alliance_in_db = commander_scores_in_db["alliance"];
    commander_scores_in_ranking = commander_scores_in_ranking["scores"];
    commander_scores_in_db = commander_scores_in_db["scores"];

    let commander_names = document.createElement("div");

    commander_names.className = "commander_names";
    commander_names.style.paddingBottom = "5px";
    commander_names.style.textAlign = "center";
    commander_names.innerHTML = "<b>" + commander_name_in_ranking + "(" + commander_alliance_in_ranking + ")</b> vs <b>" + commander_name_in_db + "(" + commander_alliance_in_db + ")</b> (" + score_diff.toFixed(2) + ")";
     if (score_diff > -0.1){
            commander_names.style.backgroundColor = "green";
        } else if (score_diff > -0.2){
            commander_names.style.backgroundColor = "yellow";
        }
        else if (score_diff > -0.3){
            commander_names.style.backgroundColor = "orange";
        }
        else{
            commander_names.style.backgroundColor = "red";
        }
    commander_score_comparison_element.appendChild(commander_names);
    let score_comparison_list = document.createElement("div");
    score_comparison_list.className = "score_comparison_list";
    commander_score_comparison_element.appendChild(score_comparison_list);
    for (let score_name in commander_scores_in_ranking) {
        if (!(displayed_rankings.has(score_name)) || !(score_name in commander_scores_in_db)) {
            continue;
        }
        let score_in_ranking = commander_scores_in_ranking[score_name];
        let score_in_db = commander_scores_in_db[score_name];
        let relative_score_diff = Math.abs(100*(score_in_ranking - score_in_db) / Math.max(score_in_db, score_in_ranking));
        let score_comparison_element = document.createElement("div");
        let score_comparison_element_color = "green";
        if (relative_score_diff > 10) {
            score_comparison_element_color = "red";
        } else if (relative_score_diff > 5) {
            score_comparison_element_color = "orange";
        } else if (relative_score_diff > 1) {
            score_comparison_element_color = "yellow";
        }

        score_comparison_element.className = "score_comparison_element";
        score_comparison_element.innerHTML = "<b>" + ranking_names[score_name] + "</b>: " +
            formatScore(score_in_ranking) + " vs " + formatScore(score_in_db) +
            " (<span style='color: " + score_comparison_element_color + ";'>" + relative_score_diff.toFixed(2) + "</span>)";
        score_comparison_list.appendChild(score_comparison_element);
    }
    for (let score_name in commander_scores_in_ranking) {
        if (!(displayed_rankings.has(score_name)) || (score_name in commander_scores_in_db)) {
            continue;
        }
        let score_in_ranking = commander_scores_in_ranking[score_name];
        let score_comparison_element = document.createElement("div");
        score_comparison_element.className = "score_comparison_element";
        score_comparison_element.innerHTML = "<b>" + ranking_names[score_name] + "</b>: " + formatScore(score_in_ranking) + " vs " + "---";
        score_comparison_list.appendChild(score_comparison_element);
    }
    for (let score_name in commander_scores_in_db) {
        if (!(displayed_rankings.has(score_name)) || (score_name in commander_scores_in_ranking)) {
            continue;
        }
        let score_in_db = commander_scores_in_db[score_name];
        let score_comparison_element = document.createElement("div");
        score_comparison_element.className = "score_comparison_element";
        score_comparison_element.innerHTML = "<b>" + ranking_names[score_name] + "</b>: " + "---" + " vs " + formatScore(score_in_db);
        score_comparison_list.appendChild(score_comparison_element);
    }
    return commander_score_comparison_element;

}

function commander_fusion_widget(fusionData, elementId) {
    let element = document.getElementById(elementId);
    let fusion = fusionData["data"];

    element.innerHTML = "<div class='commander_fusion_widget'></div>";
    let root = element.firstChild;
    element.style.marginLeft = "10px";
    let title1 = document.createElement("h3");
    title1.innerHTML = "Merged commanders";
    root.appendChild(title1);

    let matched_commanders_list = document.createElement("div");
    matched_commanders_list.className = "matched_commanders_list";
    let matched_commanders_to_insert = fusion["matched_commanders_to_insert"];
    let non_matched_commanders_db = fusion["non_matched_commanders_db"];
    for (let commander_name_in_ranking_data in matched_commanders_to_insert) {
        let commander_name_in_db = matched_commanders_to_insert[commander_name_in_ranking_data][0];
        let score_diff = matched_commanders_to_insert[commander_name_in_ranking_data][1];
        let matched_commander_element = document.createElement("div");
        matched_commander_element.className = "matched_commander_element";

        let commander_db_score = fusion["commander_db_scores"][commander_name_in_db];
        if (commander_db_score === undefined) {
            commander_db_score = {
                "alliance": "manual match",
                "scores": {}

            }
        }

        let commander_score_comparison_element = create_commander_score_comparison_element(
                commander_name_in_ranking_data, fusion["commander_to_insert_scores"][commander_name_in_ranking_data],
                commander_name_in_db, commander_db_score,
                score_diff)
        commander_score_comparison_element.style.width = "90%";
        matched_commander_element.appendChild(commander_score_comparison_element);

        matched_commanders_list.appendChild(matched_commander_element);
        let matched_commander_button = document.createElement("button");
        matched_commander_button.innerHTML = "Remove";
        matched_commander_button.onclick = async function () {
            await fusionData.remove_matched_commander(commander_name_in_ranking_data);
            commander_fusion_widget(fusionData, elementId);
        }
        matched_commander_element.appendChild(matched_commander_button);
    }
    root.appendChild(matched_commanders_list);

    let title2 = document.createElement("h3");
    title2.innerHTML = "Commanders to insert";
    root.appendChild(title2);
    let commanders_to_insert_list = document.createElement("div");
    commanders_to_insert_list.className = "commanders_to_insert_list";
    for (let commander_name of fusion["commanders_to_insert"]) {
        let close_commanders = fusion["matching_stats"][commander_name];
        if (close_commanders === undefined || close_commanders.length === 0)
            continue;

        let commander_to_insert_element = document.createElement("div");
        commander_to_insert_element.className = "commander_to_insert_element";
        commander_to_insert_element.style.verticalAlign = "top";
        for (let i = 0; i < close_commanders.length; i++) {
            let close_commander_name = close_commanders[i][1];
            let flag = false;
            for(let c2 in matched_commanders_to_insert){
                if (matched_commanders_to_insert[c2][1] === close_commander_name){
                    flag = true;
                    break;
                }
            }
            if (flag){
                continue;
            }

            let close_commander_score = close_commanders[i][0];
            let close_commander_element = create_commander_score_comparison_element(
                commander_name, fusion["commander_to_insert_scores"][commander_name],
                close_commander_name, fusion["commander_db_scores"][close_commander_name],
                close_commander_score
            );
            close_commander_element.style.width = "32%";
            close_commander_element.style.verticalAlign = "top";
            close_commander_element.style.border = "1px solid black";
            // add merge button

            let merge_button = document.createElement("button");
            merge_button.innerHTML = "Merge";
            merge_button.onclick = async function () {
                await fusionData.add_matched_commander(commander_name, close_commander_name, close_commander_score);
                commander_fusion_widget(fusionData, elementId);
            }
            close_commander_element.appendChild(merge_button);
            commander_to_insert_element.appendChild(close_commander_element);

        }

        // add a button to pick a commander name from the list
        let commander_name_button = document.createElement("button");
        commander_name_button.innerHTML = "Search commander";
        commander_name_button.onclick = async function () {
            let result = await listSelector.pickElement(non_matched_commanders_db);
            if (result !== undefined) {
                await fusionData.add_matched_commander(commander_name, result, 99);
                commander_fusion_widget(fusionData, elementId);
            }
        }
        commander_to_insert_element.appendChild(commander_name_button);

        commanders_to_insert_list.appendChild(commander_to_insert_element);
    }
    for (let commander_name of fusion["commanders_to_insert"]) {

        let close_commanders = fusion["matching_stats"][commander_name];
        if (close_commanders !== undefined && close_commanders.length > 0)
            continue;

        let commander_to_insert_element = document.createElement("div");
        commander_to_insert_element.className = "commander_to_insert_element";
        commander_to_insert_element.style.verticalAlign = "top";
        let commander_description = create_commander_description_element(commander_name, fusion["commander_to_insert_scores"][commander_name]);
        commander_description.innerHTML += "<br/><b>No close commander found</b>";
        commander_to_insert_element.appendChild(commander_description);

        // add a button to pick a commander name from the list
        let commander_name_button = document.createElement("button");
        commander_name_button.innerHTML = "Search commander";
        commander_name_button.onclick = async function () {
            let result = await listSelector.pickElement(non_matched_commanders_db);
            if (result !== undefined) {
                await fusionData.add_matched_commander(commander_name, result, 99);
                commander_fusion_widget(fusionData, elementId);
            }

        }
        commander_to_insert_element.appendChild(commander_name_button);

        commanders_to_insert_list.appendChild(commander_to_insert_element);
    }

    root.appendChild(commanders_to_insert_list);
}

function commander_ranking_widget(rankingData, elementId) {
    let element = document.getElementById(elementId);
    let ranking = rankingData["data"];
    let commandersDict = rankingData["commandersDict"];

    element.innerHTML = "<div class='commander_ranking_widget'></div>";
    let root = element.firstChild;
    let commander_list = document.createElement("div");
    commander_list.className = "commander_ranking_list";



    for (let commander of rankingData.commanders_sorted_by_warning()) {

        let commander_scores = ranking[commander];
        if (commander_scores["diagnostic"]["warning_level"] === 0) {
            continue;
        }

        let commander_ranking_element = document.createElement("div");
        commander_ranking_element.className = "commander_ranking_element";
        // first element in commander_scores
        let commander_description = document.createElement("div");
        commander_description.style.width = "30%";
        commander_description.style.display = "inline-block";
        commander_ranking_element.appendChild(commander_description);
        let score_name = Object.keys(commander_scores)[0];
        let commander_alliance_short = commander_scores[score_name]["alliance_short"];
        commander_description.innerHTML = commander + " (" + commander_alliance_short + ")";

        for (let score_name in commander_scores) {
            if (score_name === "diagnostic") {
                let diagnostic = commander_scores[score_name];
                let warning_level = diagnostic["warning_level"];
                if (warning_level === 10) {
                    commander_ranking_element.style.backgroundColor = "red";

                } else if (warning_level === 5) {
                    commander_ranking_element.style.backgroundColor = "orange";
                } else if (warning_level === 1) {
                    commander_ranking_element.style.backgroundColor = "yellow";
                } else {
                    commander_ranking_element.style.backgroundColor = "green";
                }
            } else {
                let score = commander_scores[score_name];
                commander_description.innerHTML += '<img style="height: 20px;margin-left: 5px;" src=\"data:image/png;base64,' + score["name_image"] + '" title="' + score_name + ':' + score["score"] + '(' + score["rank"] + ')"/>';
            }

        }
        let action_div = document.createElement("div");
        action_div.style.width = "70%";
        action_div.style.display = "inline-block";
        commander_ranking_element.appendChild(action_div);
        let most_similar_commanders = commander_scores["diagnostic"]["most_similars"];
        for (let i = most_similar_commanders.length - 1; i >= 0; i--) {
            let similar_commander = most_similar_commanders[i][1];
            let similar_commander_stat = most_similar_commanders[i][0];
            if (!(similar_commander in ranking)) {
                continue;
            }

            let similar_commander_scores = ranking[similar_commander];
            let similar_commander_score_name = Object.keys(similar_commander_scores)[0];
            let similar_commander_alliance_short = similar_commander_scores[similar_commander_score_name]["alliance_short"];

            let similar_commander_div = document.createElement("div");
            similar_commander_div.style.border = "1px solid black";
            let image = document.createElement("img");
            image.src = "data:image/png;base64," + similar_commander_scores[similar_commander_score_name]["name_image"];
            image.style.height = "20px";
            image.style.marginLeft = "5px";
            image.title = similar_commander_score_name + ':' + similar_commander_scores[similar_commander_score_name]["score"] + '(' + similar_commander_scores[similar_commander_score_name]["rank"] + ')';
            similar_commander_div.appendChild(image);


            let score_element = document.createElement("div");
            score_element.style.display = "inline-block";
            score_element.innerHTML = similar_commander_stat.toFixed(2);
            similar_commander_div.appendChild(score_element)


            if (similar_commander_alliance_short.toLowerCase() !== commander_alliance_short.toLowerCase()) {
                similar_commander_div.style.backgroundColor = "lightgray";
            }
            let score_details = document.createElement("div");
            score_details.id = "score_details_" + commander + "_" + similar_commander;
            score_details.style.display = "none";
            score_details.innerHTML = similar_commander + " (" + similar_commander_scores[similar_commander_score_name]["alliance_short"] + ")";
            let added_scores = [];
            let common_scores = [];
            for (let score_name in similar_commander_scores) {
                if (score_name === "diagnostic") {
                    continue;
                }
                if (score_name in commander_scores) {
                    common_scores.push(score_name);
                } else {
                    added_scores.push(score_name);
                }
            }
            if (common_scores.length > 0) {
                score_details.innerHTML += "<br/> <b>Common scores:</b> " + common_scores.join(", ");
            }
            if (added_scores.length > 0) {
                score_details.innerHTML += "<br/> <b>Added scores:</b> " + added_scores.join(", ");
            }
            similar_commander_div.appendChild(score_details);
            image.addEventListener("click", function () {
                let score_details = document.getElementById("score_details_" + commander + "_" + similar_commander);
                if (score_details.style.display === "none") {
                    score_details.style.display = "block";
                } else {
                    score_details.style.display = "none";
                }
            });

            action_div.appendChild(similar_commander_div);
            let target_commander1 = undefined;
            let other_commander1 = undefined;
            let target_commander2 = undefined;
            let other_commander2 = undefined;
            if ("commander_loss" in similar_commander_scores) {
                target_commander1 = similar_commander;
                other_commander1 = commander;
            } else if ("commander_loss" in commander_scores) {
                target_commander1 = commander;
                other_commander1 = similar_commander;
            } else {
                target_commander2 = commander;
                other_commander2 = similar_commander;
                target_commander1 = similar_commander;
                other_commander1 = commander;
            }

            let similar_commander_button_1 = document.createElement("button");
            similar_commander_button_1.innerHTML = "Merge on " + target_commander1;
            similar_commander_button_1.onclick = async function () {
                let new_name = await rankingData.merge([target_commander1, other_commander1]);
                commander_ranking_widget(rankingData, elementId);
            }
            similar_commander_div.appendChild(similar_commander_button_1);
            if (target_commander2 !== undefined) {
                let similar_commander_button_2 = document.createElement("button");
                similar_commander_button_2.innerHTML = "Merge on " + target_commander2;
                similar_commander_button_2.onclick = async function () {
                    let new_name = await rankingData.merge([target_commander2, other_commander2]);
                    commander_ranking_widget(rankingData, elementId);
                }
                similar_commander_div.appendChild(similar_commander_button_2);
            } else {
                let similar_commander_button_2 = document.createElement("button");
                similar_commander_button_2.innerHTML = "Merge and save";
                similar_commander_button_2.style.backgroundColor = "cyan";
                similar_commander_button_2.onclick = async function () {
                    await pywebview.api.ranking("save_matching_images", target_commander1, other_commander1);
                    let new_name = await rankingData.merge([target_commander1, other_commander1]);
                    commander_ranking_widget(rankingData, elementId);
                }
                similar_commander_div.appendChild(similar_commander_button_2);
            }
        }

        // add a button to pick a commander name from the list
        let commander_name_button = document.createElement("button");
        commander_name_button.innerHTML = "Search commander";
        commander_name_button.onclick = async function () {
            let result = await listSelector.pickElement(commandersDict);
            if (result !== undefined) {
                let new_name = await rankingData.merge([result, commander]);
                commander_ranking_widget(rankingData, elementId);
            }
        }
        action_div.appendChild(commander_name_button);

        commander_list.appendChild(commander_ranking_element);
    }
    root.appendChild(commander_list);

}



class listSelectorWithAutoComplete {
    constructor(rootElement) {
        this.rootElement = rootElement;
        this.nameDict = null;
        this.pendingPromise = null;
        this.init();
    }

    init() {
        // create bootstrap modal
        this.modal = document.createElement("div");
        this.modal.setAttribute("class", "modal");
        this.modal.setAttribute("tabindex", "-1");
        this.modal.setAttribute("role", "dialog");
        this.modal.style.display = "none";
        this.rootElement.appendChild(this.modal);

        this.modalDialog = document.createElement("div");
        this.modalDialog.setAttribute("class", "modal-dialog");
        this.modalDialog.setAttribute("role", "document");
        this.modal.appendChild(this.modalDialog);

        this.modalContent = document.createElement("div");
        this.modalContent.setAttribute("class", "modal-content");
        this.modalDialog.appendChild(this.modalContent);

        this.modalHeader = document.createElement("div");
        this.modalHeader.setAttribute("class", "modal-header");
        this.modalContent.appendChild(this.modalHeader);

        this.modalTitle = document.createElement("h5");
        this.modalTitle.setAttribute("class", "modal-title");
        this.modalTitle.innerHTML = "Select value";
        this.modalHeader.appendChild(this.modalTitle);

        this.closeButton = document.createElement("button");
        this.closeButton.setAttribute("type", "button");
        this.closeButton.setAttribute("class", "close");
        this.closeButton.setAttribute("data-dismiss", "modal");
        this.closeButton.setAttribute("aria-label", "Close");
        this.closeButton.innerHTML = "<span aria-hidden='true'>&times;</span>";
        this.closeButton.onclick = () => {
            this.modal.style.display = "none";
            this.resolve(undefined);
        }
        this.modalHeader.appendChild(this.closeButton);

        this.modalBody = document.createElement("div");
        this.modalBody.setAttribute("class", "modal-body");
        this.modalContent.appendChild(this.modalBody);

        this.modalFooter = document.createElement("div");
        this.modalFooter.setAttribute("class", "modal-footer");
        this.modalContent.appendChild(this.modalFooter);

        this.saveButton = document.createElement("button");
        this.saveButton.setAttribute("type", "button");
        this.saveButton.setAttribute("class", "btn btn-primary");
        this.saveButton.innerHTML = "Save";
        this.saveButton.onclick = () => {
            this.modal.style.display = "none";
            this.resolve(this.input_field.getAttribute("data-key"));
        }

        this.cancelButton = document.createElement("button");
        this.cancelButton.setAttribute("type", "button");
        this.cancelButton.setAttribute("class", "btn btn-secondary");
        this.cancelButton.innerHTML = "Cancel";
        this.cancelButton.onclick = () => {
            this.modal.style.display = "none";
            this.resolve(undefined);
        }

        this.modalFooter.appendChild(this.saveButton);
        this.modalFooter.appendChild(this.cancelButton);

        this.input_field = document.createElement("input");
        this.input_field.setAttribute("type", "text");
        this.input_field.setAttribute("id", "commander_input");
        this.input_field.setAttribute("class", "commander_input");
        this.input_field.setAttribute("placeholder", "Commander name");
        this.input_field.setAttribute("style", "width: 100%;");
        this.input_field.style.width = "70%";
        this.modalBody.appendChild(this.input_field);
        this.button_clear = document.createElement("button");
        this.button_clear.innerHTML = "Clear";
        this.button_clear.style.display = "none";
        this.button_clear.onclick = () => {
            this.input_field.value = "";
            this.button_clear.style.display = "none";
        }
        this.modalBody.appendChild(this.button_clear);

        document.addEventListener("click", (function (e) {
            this.closeAllLists(e.target);
        }).bind(this));

        this._initAutocomplete();
    }

    _initAutocomplete() {
         this.input_field.addEventListener("keydown", (function (e) {
                const key = e.code || e.keyCode;
                if (key === 'Enter' || key === 13) {
                    this.input_field.blur();
                    /*If the ENTER key is pressed, prevent the form from being submitted,*/
                    e.preventDefault();
                }
            }).bind(this));

            this.input_field.addEventListener("keyup", (function (e) {
                const key = e.code || e.keyCode;
                if (key === 'Enter' || key === 13) {
                    this.input_field.blur();
                    /*If the ENTER key is pressed, prevent the form from being submitted,*/
                    e.preventDefault();
                }
            }).bind(this));

            this.input_field.addEventListener("keypress", (function (e) {
                const key = e.code || e.keyCode;
                if (key === 'Enter' || key === 13) {
                    this.input_field.blur();
                    /*If the ENTER key is pressed, prevent the form from being submitted,*/
                    e.preventDefault();
                }
            }).bind(this));

            this.input_field.addEventListener("change", (function (e) {
                this.input_field.focus();
                this.input_field.blur();
                e.preventDefault();
                return false;
            }).bind(this));

            this.input_field.addEventListener("input", (function (e) {
                var val = this.input_field.value;
                this.closeAllLists();
                if (!val) {
                    this.button_clear.style.display = "none";
                    //button_search.style.display = "inline";
                    return false;
                } else {
                    //button_search.style.display = "none";
                    this.button_clear.style.display = "inline";
                }
                this.currentFocus = -1;
                /*create a DIV element that will contain the items (values):*/
                var a = document.createElement("DIV");
                //a.setAttribute("id", elem.id + "autocomplete-list");
                a.setAttribute("class", "autocomplete-items");
                /*append the DIV element as a child of the autocomplete container:*/
                this.input_field.parentNode.appendChild(a);
                val = val.toLowerCase();
                //const foundNames = new Set();

                let searchFunction = function () {

                    let boldise = function (str, index, length) {
                        return str.substring(0, index) + "<b>" + str.substring(index, index + length) + "</b>" + str.substring(index + length);
                    }

                    let addFoundElement = function (boldisedName, key) {
                        /*create a DIV element for each matching element:*/
                        let b = document.createElement("div");
                        b.innerHTML = boldisedName;
                        /*insert a input field that will hold the current array item's value:*/
                        b.innerHTML += "<input type='hidden' value='" + key + "'>";
                        /*execute a function when someone clicks on the item value (DIV element):*/
                        b.addEventListener("click", function (e) {
                            /*insert the value for the autocomplete text field:*/
                            this.input_field.value = this.nameDict[key]["name"];
                            this.input_field.setAttribute("data-key", key);
                            this.button_clear.style.display = "none";

                            /*close the list of autocompleted values,
                            (or any other open lists of autocompleted values:*/
                            this.closeAllLists();
                        }.bind(this));
                        a.appendChild(b);
                    }.bind(this);

                    /* first test names*/
                    for (const [key, value] of Object.entries(this.nameDict)) {

                        let name = value["name"];
                        let pos = name.toLowerCase().indexOf(val);
                        /*check if the item starts with the same letters as the text field value:*/
                        if (pos != -1) {
                            //foundNames.add(key);
                            addFoundElement(boldise(name, pos, val.length), key);
                        }
                    }

                }.bind(this);
                searchFunction();


            }).bind(this));
    }



    async pickElement(nameDict) {
        this.nameDict = nameDict;
        this.modal.style.display = "block";
        this.input_field.focus();
        this.input_field.value = "";
        this.button_clear.style.display = "none";
        this.pendingPromise = new Promise((resolve, reject) => {
            this.resolve = resolve;
            this.reject = reject;
        });
        return this.pendingPromise;
    }

    closeAllLists(elmnt=null) {
        /*close all autocomplete lists in the document, except the one passed as an argument:*/
        var x = document.getElementsByClassName("autocomplete-items");
        for (var i = 0; i < x.length; i++) {
            if (elmnt != x[i] && elmnt != this.input_field) {
                x[i].parentNode.removeChild(x[i]);
            }
        }
    }

}

var listSelector = new listSelectorWithAutoComplete(document.body);

/*
var testNames = {
    "toto": { "name": "toto (DCI)", "alliance": "DCI" },
    "tata": { "name": "tata (DCI)", "alliance": "DCI" },
    "titi": { "name": "titi (DCI)", "alliance": "DCI" },
    "tutu": { "name": "tutu (DAY)", "alliance": "DCI" },
    "tete": { "name": "tete (DAY)", "alliance": "DCI" },
}

async function testListSelector() {
    let result = await listSelector.pickElement(testNames);
    console.log(result);
}

testListSelector();*/
