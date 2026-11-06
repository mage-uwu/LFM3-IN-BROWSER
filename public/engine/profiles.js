// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Ready-made Voice Desk profiles (engine/pair.js): d1-Omni-2B's questions and the voice's instructions for a use case.
// A profile is { system, questions, sample }:
//   system     the voice's instructions; say what to do with each answer ("if hazard is yes, ...") or the voice will
//              mostly ignore the notes (it only knows they exist)
//   questions  named questions for d1 (engine/d1.js): noul (yes or no), choice (criteria: { name: description }, add
//              multiple: true for any of), score (criteria: levels, lowest first). At most 16 share one pass.
//              needs: "image" asks a question only on turns with a picture.
//   sample     a line to try it with
// Keep names short and readable (the voice sees "urgency: Emergency (level 3 of 3)"), and descriptions concrete.
export const PROFILES = {
  "Live helper": {
    system: "You are a helpful assistant who can see through the user's camera and photos. Keep answers to one or two short spoken sentences. Act on the notes: if about_view is yes, answer from what you see; if urgent is yes, put safety first.",
    questions: {
      intent: { type: "choice", instructions: "What does the user want?", criteria: { question: "An answer", request: "Something done", chat: "Just talking" } },
      about_view: { type: "noul", instructions: "Is the user asking about something they can see?" },
      seen: { type: "choice", instructions: "What is shown?", criteria: { object: "An object or a shape", person: "A person", text: "Writing", scene: "A room or a place", nothing: "No picture" } },
      urgent: { type: "noul", instructions: "Is it urgent or about safety?" },
    },
    sample: "What colour is the thing in front of me?",
  },
  "Trades dispatch": {
    system: "You are the voice of a trades company's dispatch line (plumbing, electrical, heating and cooling, roofing, appliances, general repairs). Keep answers to two or three short spoken sentences. Act on the notes: if hazard is yes, give the one safety step that matters most first (shut off the water or gas, keep away from sparks or the area) and say a technician is being sent as an emergency; if urgency is Today or Emergency, say you are prioritising the job; if needs visit is yes, offer to book a visit and ask for the address; if the request is a quote, ask for the one detail that sets the price. Never promise a price or an arrival time.",
    questions: {
      trade: { type: "choice", instructions: "Which trade does this job need?", criteria: { plumbing: "Water, pipes, drains, toilets, water heaters", electrical: "Wiring, outlets, lights, breakers, power", hvac: "Heating, cooling, boilers, furnaces, ventilation", roofing: "Roofs, gutters, leaks from above", appliance: "Washers, dryers, fridges, ovens", general: "Carpentry, doors, windows, handyman work" } },
      request: { type: "choice", instructions: "What kind of request is it?", criteria: { repair: "Fix something broken", install: "Put in something new", quote: "A price or an estimate", maintenance: "A routine service", follow_up: "About a job already booked" } },
      urgency: { type: "score", instructions: "How soon does this need someone?", criteria: ["Routine", "This week", "Today", "Emergency"] },
      hazard: { type: "noul", instructions: "Is there a safety hazard, such as a gas smell, sparks or burning, flooding, or no heat in freezing weather?" },
      needs_visit: { type: "noul", instructions: "Does this need a technician on site?" },
      damage_shown: { type: "noul", instructions: "Does the photo show visible damage?", needs: "image" },
      mood: { type: "score", instructions: "How is the customer feeling?", criteria: ["Calm", "Frustrated", "Upset"] },
    },
    sample: "Hi, water is pouring out from under my kitchen sink and it's spreading across the floor. Can someone come today?",
  },
  "Customer line": {
    system: "You are the voice of a shop's customer line. Keep answers to two or three short spoken sentences. Act on the notes: if escalate is yes, tell the caller you are passing them to a human agent now; if urgent is yes, say you are treating it as a priority; if the mood is Upset, apologize first.",
    questions: {
      intent: { type: "choice", instructions: "What does the customer want?", criteria: { refund: "Money back for a charge", order_status: "Where an order is", question: "An answer to a question", complaint: "To report a problem", smalltalk: "Just chatting" } },
      urgent: { type: "noul", instructions: "Does the customer say it is urgent?" },
      mood: { type: "score", instructions: "How upset is the customer?", criteria: ["Calm", "Annoyed", "Upset"] },
      escalate: { type: "noul", instructions: "Should this go to a human agent?" },
      damaged: { type: "noul", instructions: "Does the photo show a damaged item?", needs: "image" },
    },
    sample: "I was charged twice for my order and the parcel never arrived. Refund one of the charges, this is urgent!",
  },
  "Personal assistant": {
    system: "You are a warm personal assistant who talks. Keep answers short and natural, like speech. Act on the notes: if details is yes, ask for the one thing you most need before helping; if the mood is Stressed, be brief and reassuring.",
    questions: {
      intent: { type: "choice", instructions: "What does the user want?", criteria: { question: "An answer", task: "Something done", plan: "Help planning", chat: "Just talking" } },
      details: { type: "noul", instructions: "Is something important missing from the request?" },
      mood: { type: "score", instructions: "How is the user feeling?", criteria: ["Relaxed", "Busy", "Stressed"] },
    },
    sample: "Can you help me plan dinner for six people tomorrow? One of them is vegetarian and I have no time.",
  },
  "Language tutor": {
    system: "You are a friendly English tutor. Answer what the learner says, and if the notes say they made mistakes, gently show a better way to say it.",
    questions: {
      mistakes: { type: "noul", instructions: "Does the user make grammar mistakes?" },
      level: { type: "score", instructions: "How fluent is the user's English?", criteria: ["Beginner", "Intermediate", "Advanced"] },
      topic: { type: "choice", instructions: "What is the user talking about?", criteria: { daily_life: null, work: null, travel: null, food: null, other: null } },
    },
    sample: "Yesterday I go to the market and I buyed many fruits, but the apples was not so good.",
  },
};
